import os
import numpy as np
from scipy import signal
from scipy.io import loadmat

try:
    from sklearn.ensemble import RandomForestClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class MovementMismatchError(Exception):
    """Exception raised when detected activations do not match the protocol vector."""
    def __init__(self, message: str, detected_count: int = 0, expected_count: int = 0, aux_plot_data: dict = None):
        super().__init__(message)
        self.detected_count = detected_count
        self.expected_count = expected_count
        self.aux_plot_data = aux_plot_data or {}


def parse_protocol_string(protocol_str: str) -> np.ndarray:
    """Converts a string like '1,1,0,0,1,1,0,0' (or '1 1 0 0') into a 1D numpy array of 0s and 1s."""
    cleaned = protocol_str.replace(';', ',').replace(' ', ',')
    tokens = [t.strip() for t in cleaned.split(',') if t.strip()]
    if not tokens:
        raise ValueError("Protocol sequence is empty. Please enter a sequence of 0s and 1s (e.g. 1,1,0,0,1,1,0,0).")
    values = [int(t) for t in tokens]
    if any(v not in (0, 1) for v in values):
        raise ValueError("Protocol sequence must contain only 0 (flexion/FCR) and 1 (extension/ECR).")
    return np.array(values, dtype=int)


def _to_scalar_float(val, default=2000.0) -> float:
    """Safely converts numpy arrays/scalars or numeric inputs to a standard Python float."""
    if val is None:
        return float(default)
    if isinstance(val, (np.ndarray, list)):
        arr = np.squeeze(val)
        return float(arr.item() if arr.ndim == 0 else arr[0])
    return float(val)


def detect_bursts(env_mean: np.ndarray, threshold: float, fs: float, min_gap_ms: float = 200.0, min_dur_ms: float = 150.0):
    """Detects activation bursts (onsets and offsets) from the mean envelope."""
    fs = _to_scalar_float(fs)
    threshold = _to_scalar_float(threshold, default=0.0)

    is_above = (env_mean >= threshold).astype(int)
    diff = np.diff(np.pad(is_above, (1, 1), 'constant'))

    onsets = np.where(diff == 1)[0]
    offsets = np.where(diff == -1)[0]

    if len(onsets) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    min_gap_samples = int(np.round(min_gap_ms * fs / 1000.0))
    min_dur_samples = int(np.round(min_dur_ms * fs / 1000.0))

    # Merge bursts separated by gaps shorter than min_gap
    merged_onsets, merged_offsets = [onsets[0]], []
    for i in range(len(onsets) - 1):
        if onsets[i + 1] - offsets[i] < min_gap_samples:
            continue
        merged_offsets.append(offsets[i])
        merged_onsets.append(onsets[i + 1])
    merged_offsets.append(offsets[-1])

    # Discard bursts shorter than min_dur
    final_onsets, final_offsets = [], []
    for on, off in zip(merged_onsets, merged_offsets):
        if (off - on) >= min_dur_samples:
            final_onsets.append(on)
            final_offsets.append(off)

    return np.array(final_onsets, dtype=int), np.array(final_offsets, dtype=int)


def run_read_calibration(emg_raw_16ch: np.ndarray, protocol: np.ndarray, fs: float = 2000.0):
    """
    Executes EMG Read Calibration:
    1. Bandpass filters (20-450 Hz) all 16 channels.
    2. Computes 16-channel RMS envelope (200 ms window) and mean envelope.
    3. Detects activation bursts and compares count against len(protocol).
    4. Ranks channels by activation energy for ECR (1) and FCR (0), returning:
       - ecr_ch (int): best 0-based channel index for ECR
       - fcr_ch (int): best 0-based channel index for FCR
       - aux_plot_data (dict): raw 16ch, filtered 16ch, envelope, threshold, bursts, and Top 3 ECR/FCR indices.
    """
    fs = _to_scalar_float(fs)
    num_channels, n_samples = emg_raw_16ch.shape

    if n_samples <= 30:
        raise ValueError("Recorded EMG signal is too short for filtering. Please record for at least 1 second.")

    # 1. Bandpass filter 20-450 Hz
    nyq = fs / 2.0
    low = max(0.001, min(20.0 / nyq, 0.98))
    high = max(low + 0.01, min(450.0 / nyq, 0.98))
    b, a = signal.butter(4, [low, high], btype='bandpass')
    emg_filt = signal.filtfilt(b, a, emg_raw_16ch, axis=1)

    # 2. RMS Envelope per channel & mean envelope
    smooth_n = max(1, int(np.round(200.0 * fs / 1000.0)))
    kernel = np.ones(smooth_n) / smooth_n
    env_all = np.zeros_like(emg_filt)
    for ch in range(num_channels):
        env_all[ch, :] = np.sqrt(np.convolve(emg_filt[ch, :] ** 2, kernel, mode='same'))

    env_mean = np.mean(env_all, axis=0)
    expected_count = int(len(protocol))
    threshold =  0.7* (_find_optimal_threshold(env_mean, expected_count, fs))

    # 3. Burst detection & protocol count check
    onsets, offsets = detect_bursts(env_mean, threshold, fs)
    detected_count = int(len(onsets))
    expected_count = int(len(protocol))

    time_vec = np.linspace(0, n_samples / fs, n_samples)

    if detected_count != expected_count:
        partial_aux = {
            'time': time_vec,
            'emg_raw_16ch': emg_raw_16ch,
            'emg_filt_16ch': emg_filt,
            'env_mean': env_mean,
            'threshold': threshold,
            'onsets': onsets / fs,
            'offsets': offsets / fs,
            'protocol': protocol,
            'top3_ecr': [],
            'top3_fcr': [],
            'ecr_ch': None,
            'fcr_ch': None,
        }
        raise MovementMismatchError(
            f"Activation count mismatch: detected {detected_count} movement(s), "
            f"but protocol specifies {expected_count} movement(s). Please repeat the recording.",
            detected_count=detected_count,
            expected_count=expected_count,
            aux_plot_data=partial_aux
        )

    # 4. Compute channel activation energies for ECR (1) and FCR (0)
    ch_energies_ecr = np.zeros(num_channels)
    ch_energies_fcr = np.zeros(num_channels)

    for i, (on, off) in enumerate(zip(onsets, offsets)):
        trial_type = protocol[i]
        burst_energy = np.sum(env_all[:, on:off] ** 2, axis=1)
        if trial_type == 1:
            ch_energies_ecr += burst_energy
        else:
            ch_energies_fcr += burst_energy

    # Rank channels descending by energy (0-based indices)
    ranked_ecr = np.argsort(ch_energies_ecr)[::-1]
    ranked_fcr = np.argsort(ch_energies_fcr)[::-1]

    top3_ecr = [int(ch) for ch in ranked_ecr[:3]]
    top3_fcr = [int(ch) for ch in ranked_fcr[:3]]

    ecr_ch = top3_ecr[0]
    fcr_ch = top3_fcr[0]

    # Ensure primary ECR and FCR channels are distinct
    if ecr_ch == fcr_ch and len(ranked_fcr) > 1:
        fcr_ch = int(ranked_fcr[1])

    aux_plot_data = {
        'time': time_vec,
        'emg_raw_16ch': emg_raw_16ch,
        'emg_filt_16ch': emg_filt,
        'env_mean': env_mean,
        'threshold': threshold,
        'onsets': onsets / fs,
        'offsets': offsets / fs,
        'protocol': protocol,
        'ch_energies_ecr': ch_energies_ecr,
        'ch_energies_fcr': ch_energies_fcr,
        'top3_ecr': top3_ecr,
        'top3_fcr': top3_fcr,
        'ecr_ch': ecr_ch,
        'fcr_ch': fcr_ch,
    }

    return ecr_ch, fcr_ch, aux_plot_data


# Alias di compatibilità
select_read_channels = run_read_calibration


def _find_optimal_threshold(env_mean: np.ndarray, expected_count: int, fs: float, n_steps: int = 35) -> float:
    """
    Finds the most stable burst threshold within a noise-guarded range such that
    detect_bursts() returns exactly `expected_count` activations.
    Falls back to a baseline-anchored heuristic if no threshold matches.
    """
    # 1. Estimate resting baseline and robust noise floor from the lower 50% of samples
    lower_half = env_mean[env_mean <= np.percentile(env_mean, 50)]
    if len(lower_half) > 0:
        baseline_med = float(np.median(lower_half))
        mad = float(np.median(np.abs(lower_half - baseline_med)))
        # 1.4826 scales MAD to standard deviation for Gaussian noise
        noise_floor = baseline_med + 5.0 * (1.4826 * mad + 1e-6)
    else:
        noise_floor = float(np.percentile(env_mean, 25))

    high_ref = float(np.percentile(env_mean, 95))
    min_thresh = max(noise_floor, 0.05 * high_ref)
    max_thresh = max(min_thresh * 1.2, 0.75 * high_ref)

    default_thresh = min_thresh + 0.25 * (max_thresh - min_thresh)
    if expected_count <= 0 or max_thresh <= min_thresh:
        return default_thresh

    # 2. Evaluate candidate thresholds across the safe range
    candidates = np.linspace(min_thresh, max_thresh, n_steps)
    valid_mask = np.zeros(n_steps, dtype=bool)

    for idx, th in enumerate(candidates):
        onsets, _ = detect_bursts(env_mean, float(th), fs)
        if len(onsets) == expected_count:
            valid_mask[idx] = True

    if not np.any(valid_mask):
        return default_thresh

    # 3. Pick the midpoint of the widest contiguous plateau of valid thresholds
    padded = np.pad(valid_mask.astype(int), (1, 1), 'constant')
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]  # exclusive

    best_run = int(np.argmax(ends - starts))
    plateau_candidates = candidates[starts[best_run]:ends[best_run]]
    return float(np.median(plateau_candidates))


# =====================================================================
#  2. STIMULATION CALIBRATION (ML CLASSIFIER + NEIGHBOR FALLBACK)
# =====================================================================

def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Compatibility wrapper for trapezoidal integration across NumPy 1.x and 2.x."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


class EvokedResponseClassifier:
    """
    Loads `classificatore_py.mat` exported from MATLAB and scores evoked epochs
    using Random Forest and Linear SVM with identical standardization.
    """
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "classificatore_py.mat")
        self.model_path = model_path
        self.loaded = False
        self.features = ['quota_risposta', 'snr_n', 'pp_amp_n', 'rapporto_80']
        self.da_normalizzare = ['snr', 'pp_amp']
        self.mu = np.zeros(4)
        self.sg = np.ones(4)
        self.rf_soglia = 0.5
        self.sv_soglia = 0.0
        self.sv_beta = None
        self.sv_bias = 0.0
        self.rf_model = None

        if os.path.isfile(self.model_path):
            self._load_from_mat(self.model_path)

    def _load_from_mat(self, path: str):
        data = loadmat(path, squeeze_me=True)
        if 'features' in data:
            self.features = [str(f) for f in np.atleast_1d(data['features'])]
        if 'da_normalizzare' in data:
            self.da_normalizzare = [str(f) for f in np.atleast_1d(data['da_normalizzare'])]
        self.mu = np.asarray(data['mu'], dtype=float).ravel()
        self.sg = np.asarray(data['sg'], dtype=float).ravel()
        self.sg[self.sg == 0] = 1.0
        self.rf_soglia = float(data.get('rf_soglia', 0.5))
        self.sv_soglia = float(data.get('sv_soglia', 0.0))
        self.sv_beta = np.asarray(data['sv_beta'], dtype=float).ravel()
        self.sv_bias = float(data['sv_bias'])

        if HAS_SKLEARN and 'X_train' in data and 'y_train' in data:
            X_tr = np.asarray(data['X_train'], dtype=float)
            y_tr = np.asarray(data['y_train'], dtype=int).ravel()
            Xs_tr = (X_tr - self.mu) / self.sg
            self.rf_model = RandomForestClassifier(
                n_estimators=300,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            self.rf_model.fit(Xs_tr, y_tr)
        self.loaded = True

    def predict_epochs(self, feature_matrix: np.ndarray, model_type: str = 'rf'):
        """
        Scores rows of `feature_matrix` (n_epochs x n_features).
        Returns (pred_bool, score_rf, score_sv).
        """
        n_ep = feature_matrix.shape[0]
        score_rf = np.full(n_ep, np.nan)
        score_sv = np.full(n_ep, np.nan)
        pred = np.zeros(n_ep, dtype=bool)

        if not self.loaded:
            # Fallback heuristic if classificatore_py.mat is not yet exported
            snr_n = feature_matrix[:, 1]
            quota = feature_matrix[:, 0]
            score_rf = np.clip((snr_n - 1.0) / 3.0, 0.0, 1.0)
            score_sv = snr_n - 1.8
            pred = (snr_n > 1.8) & (quota > 0.20)
            return pred, score_rf, score_sv

        Xs = (feature_matrix - self.mu) / self.sg
        finite_mask = np.all(np.isfinite(Xs), axis=1)
        if not np.any(finite_mask):
            return pred, score_rf, score_sv

        Xf = Xs[finite_mask]
        if self.sv_beta is not None:
            score_sv[finite_mask] = Xf @ self.sv_beta + self.sv_bias

        if self.rf_model is not None:
            probs = self.rf_model.predict_proba(Xf)
            col_valid = list(self.rf_model.classes_).index(1)
            score_rf[finite_mask] = probs[:, col_valid]
        elif self.sv_beta is not None:
            score_rf[finite_mask] = 1.0 / (1.0 + np.exp(-score_sv[finite_mask]))

        if model_type.lower() == 'sv':
            pred[finite_mask] = score_sv[finite_mask] >= self.sv_soglia
        else:
            pred[finite_mask] = score_rf[finite_mask] >= self.rf_soglia

        return pred, score_rf, score_sv


_CACHED_CLASSIFIER = None


def get_evoked_classifier(model_path: str = None) -> EvokedResponseClassifier:
    global _CACHED_CLASSIFIER
    if _CACHED_CLASSIFIER is None or model_path is not None:
        _CACHED_CLASSIFIER = EvokedResponseClassifier(model_path=model_path)
    return _CACHED_CLASSIFIER


def calcola_features(ep: np.ndarray, eb: np.ndarray, fs: float) -> dict:
    """Exact Python port of MATLAB `calcola_features.m`."""
    eps = np.finfo(float).eps
    ep = np.asarray(ep, dtype=float).ravel()
    eb = np.asarray(eb, dtype=float).ravel()

    ep_zm = ep - np.mean(ep)
    eb_zm = eb - np.mean(eb)

    f, pxx = signal.periodogram(ep_zm, fs=fs, window='hamming', nfft=4096, scaling='density')

    n_lev = 5
    exponents = np.arange(n_lev + 1, 0, -1, dtype=float)
    bande = np.concatenate(([0.0], fs / (2.0 ** exponents)))
    E = np.zeros(len(bande) - 1, dtype=float)
    for k in range(len(E)):
        m = (f >= bande[k]) & (f < bande[k + 1])
        if np.sum(m) >= 2:
            E[k] = _trapz(pxx[m], f[m])

    # MATLAB E(3:4) -> Python indices 2 and 3 (62.5-125 Hz and 125-250 Hz at 2000 Hz)
    quota_risposta = float((E[2] + E[3]) / (np.sum(E) + eps))

    rms_ep = float(np.sqrt(np.mean(ep_zm ** 2)))
    rms_eb = float(np.sqrt(np.mean(eb_zm ** 2)))
    snr = rms_ep / (rms_eb + eps)

    pp_amp = float(np.max(ep) - np.min(ep))

    f_taglio = 80.0
    m_alta = f >= f_taglio
    m_bassa = f < f_taglio
    P_alta = _trapz(pxx[m_alta], f[m_alta]) if np.sum(m_alta) >= 2 else 0.0
    P_bassa = _trapz(pxx[m_bassa], f[m_bassa]) if np.sum(m_bassa) >= 2 else 0.0
    rapporto_80 = float(np.log10((P_alta + eps) / (P_bassa + eps)))

    return {
        'quota_risposta': quota_risposta,
        'snr': snr,
        'pp_amp': pp_amp,
        'rapporto_80': rapporto_80,
    }


def normalizza_fondo(x: np.ndarray) -> np.ndarray:
    """Exact Python port of MATLAB `normalizza_fondo.m` (divides by 25th percentile)."""
    eps = np.finfo(float).eps
    x = np.asarray(x, dtype=float).ravel()
    ok = np.isfinite(x)
    if np.sum(ok) < 2:
        return x.copy()
    xs = np.sort(x[ok])
    idx = max(0, int(np.floor(0.25 * len(xs) + 0.5)) - 1)
    rif = xs[idx]
    return x / (rif + eps)


def neighbors_of(ch_0based: int, n_ring: int = 16, exclude: list = None) -> list:
    """Returns 0-based neighbor channels (ch-1, ch+1) on a ring of `n_ring` channels."""
    exclude_set = set(exclude or [])
    raw_nb = [(ch_0based - 1) % n_ring, (ch_0based + 1) % n_ring]
    out = []
    for nb in raw_nb:
        if nb not in exclude_set and nb not in out:
            out.append(nb)
    return out


def detect_stim_artifacts(emg_raw_16ch: np.ndarray, fs: float, isi_s: float) -> np.ndarray:
    """
    Detects stimulation artifact onset indices (0-based samples) matching the MATLAB script.
    """
    n_ch, n_samples = emg_raw_16ch.shape
    sig_diff_all = np.zeros((n_samples, n_ch), dtype=float)
    for ch in range(n_ch):
        d = np.abs(np.diff(emg_raw_16ch[ch, :]))
        sig_diff_all[:-1, ch] = d
    sig_diff = np.mean(sig_diff_all, axis=1)

    max_diff = float(np.max(sig_diff))
    if max_diff <= 0:
        return np.array([], dtype=int)

    refractory_ms = max(200.0, 0.6 * isi_s * 1000.0)
    refractory_samp = max(1, int(np.round(refractory_ms / 1000.0 * fs)))

    peaks, _ = signal.find_peaks(
        sig_diff,
        height=max_diff * 0.17,
        distance=refractory_samp
    )

    look_back_samp = int(np.round(0.010 * fs))
    head_len = min(1000, len(sig_diff))
    noise_floor_diff = float(np.median(sig_diff) + 3.0 * np.std(sig_diff[:head_len], ddof=1))

    stim_idx = np.zeros(len(peaks), dtype=int)
    for s, pk in enumerate(peaks):
        a = max(0, pk - look_back_samp)
        seg = sig_diff[a:pk + 1]
        below = np.where(seg < noise_floor_diff)[0]
        stim_idx[s] = (a + int(below[-1])) if len(below) > 0 else a

    return stim_idx


def evaluate_stim_round(
    emg_raw_16ch: np.ndarray,
    target_ch: int,
    other_main_ch: int,
    active_electrodes_1based: list,
    current_ma: float,
    n_giri: int,
    fs: float = 2000.0,
    isi_s: float = 1.0,
    role_name: str = "Extensor",
    model_type: str = 'rf',
    min_valid_rounds: int = 1,
    min_valid_fraction: float = 0.5,
    criterion: str = 'pp',
    use_neighbors: bool = True,
    n_ring: int = 16,
    skip_first_mode: str = 'auto',
    model_path: str = None,
) -> dict:
    """
    Evaluates ONE stimulation round at fixed `current_ma` across `active_electrodes_1based`
    (repeated `n_giri` times) for a single target nerve/muscle (`target_ch`).

    Returns a dictionary containing:
      - 'found': bool
      - 'chosen_el': int or None (1-based electrode number)
      - 'chosen_ch': int or None (0-based EMG channel index, main or neighbor)
      - 'from_neighbor': bool
      - 'current_ma': float
      - 'used_neighbors': bool
      - 'message': str
      - 'grid_data': dict with epochs (5-30 ms), scores, predictions, pp, and raw trace info for Assistive Plots
    """
    fs = _to_scalar_float(fs)
    n_ch_tot, n_samples = emg_raw_16ch.shape
    n_elec_active = len(active_electrodes_1based)

    nb_channels = neighbors_of(target_ch, n_ring=n_ring, exclude=[target_ch, other_main_ch]) if use_neighbors else []
    analysed_channels = [target_ch] + nb_channels
    is_main_list = [True] + [False] * len(nb_channels)

    win_samp = (int(np.round(5.0 / 1000.0 * fs)), int(np.round(30.0 / 1000.0 * fs)))
    noise_win_samp = (int(np.round(-305.0 / 1000.0 * fs)), int(np.round(-55.0 / 1000.0 * fs)))
    n_idx = np.arange(win_samp[0], win_samp[1] + 1)
    n_idx_noise = np.arange(noise_win_samp[0], noise_win_samp[1] + 1)
    t_win_ms = n_idx / fs * 1000.0

    stim_idx = detect_stim_artifacts(emg_raw_16ch, fs=fs, isi_s=isi_s)
    n_tot = len(stim_idx)
    if n_tot == 0:
        return {
            'found': False,
            'chosen_el': None,
            'chosen_ch': None,
            'from_neighbor': False,
            'current_ma': current_ma,
            'used_neighbors': False,
            'message': f"{role_name}: No stimulation artifacts detected at {current_ma:.1f} mA.",
            'grid_data': None,
        }

    # Check if EAST skipped first stimulus (El.1, giro 1)
    isi_det = np.diff(stim_idx) / fs
    gap_mid = bool(np.any(isi_det > 1.5 * isi_s)) if len(isi_det) > 0 else False
    if skip_first_mode == 'on':
        first_skipped = True
    elif skip_first_mode == 'off':
        first_skipped = False
    else:
        one_short = (n_elec_active > 1) and (n_tot % n_elec_active == n_elec_active - 1)
        first_skipped = one_short and (not gap_mid)

    off = 1 if first_skipped else 0
    s_indices = np.arange(n_tot)
    el_local_of = (s_indices + off) % n_elec_active
    el_1based_of = np.array([active_electrodes_1based[idx] for idx in el_local_of], dtype=int)
    giro_of = ((s_indices + off) // n_elec_active) + 1
    n_giri_det = max(n_giri, int(np.max(giro_of)))

    valid_mask = (stim_idx + win_samp[1]) < n_samples
    baseline_ok = (stim_idx + noise_win_samp[0]) >= 0
    usable_idx = np.where(valid_mask)[0]

    if len(usable_idx) == 0:
        return {
            'found': False,
            'chosen_el': None,
            'chosen_ch': None,
            'from_neighbor': False,
            'current_ma': current_ma,
            'used_neighbors': False,
            'message': f"{role_name}: Recording ended before post-stimulus response windows.",
            'grid_data': None,
        }

    # Extract response (5-30 ms) and baseline (-305..-55 ms) epochs
    n_analysed = len(analysed_channels)
    epochs = np.full((n_analysed, n_tot, len(n_idx)), np.nan)
    baselines = np.full((n_analysed, n_tot, len(n_idx_noise)), np.nan)

    for k, ch in enumerate(analysed_channels):
        sig_ch = emg_raw_16ch[ch, :]
        for s in usable_idx:
            epochs[k, s, :] = sig_ch[stim_idx[s] + n_idx]
            if baseline_ok[s]:
                baselines[k, s, :] = sig_ch[stim_idx[s] + n_idx_noise]

    # Replace missing early baselines with first complete baseline
    valid_with_base = np.where(valid_mask & baseline_ok)[0]
    if len(valid_with_base) > 0:
        ref_s = valid_with_base[0]
        for s in np.where(valid_mask & ~baseline_ok)[0]:
            baselines[:, s, :] = baselines[:, ref_s, :]
    else:
        for k, ch in enumerate(analysed_channels):
            for s in usable_idx:
                end_b = max(10, stim_idx[s] - int(0.005 * fs))
                baselines[k, s, :] = np.pad(
                    emg_raw_16ch[ch, :end_b],
                    (0, max(0, len(n_idx_noise) - end_b)),
                    mode='edge'
                )[:len(n_idx_noise)]

    # Feature extraction, within-channel baseline normalization, and ML scoring
    clf = get_evoked_classifier(model_path=model_path)
    score_rf = np.full((n_analysed, n_tot), np.nan)
    score_sv = np.full((n_analysed, n_tot), np.nan)
    pred = np.zeros((n_analysed, n_tot), dtype=bool)
    pp = np.full((n_analysed, n_tot), np.nan)

    for k in range(n_analysed):
        feat_dict_list = []
        for s in usable_idx:
            ep = epochs[k, s, :]
            eb = baselines[k, s, :]
            fi = calcola_features(ep, eb, fs)
            feat_dict_list.append(fi)
            pp[k, s] = float(np.max(ep) - np.min(ep))

        cols = {}
        for base_name in ['quota_risposta', 'snr', 'pp_amp', 'rapporto_80']:
            cols[base_name] = np.array([fd[base_name] for fd in feat_dict_list], dtype=float)
        for norm_base in clf.da_normalizzare:
            if norm_base in cols:
                cols[f"{norm_base}_n"] = normalizza_fondo(cols[norm_base])

        X_k = np.column_stack([cols[fname] for fname in clf.features])
        pred_k, s_rf_k, s_sv_k = clf.predict_epochs(X_k, model_type=model_type)
        pred[k, usable_idx] = pred_k
        score_rf[k, usable_idx] = s_rf_k
        score_sv[k, usable_idx] = s_sv_k

    score_active = score_sv if model_type.lower() == 'sv' else score_rf

    def _find_candidates_on_channel(k_idx: int) -> list:
        cands = []
        for el_1b in active_electrodes_1based:
            s_el = np.where((el_1based_of == el_1b) & valid_mask)[0]
            if len(s_el) == 0:
                continue
            v_el = pred[k_idx, s_el]
            n_v = int(np.sum(v_el))
            req_valid = max(min_valid_rounds, int(np.ceil(min_valid_fraction * len(s_el))))
            if n_v >= req_valid:
                pp_val = float(np.mean(pp[k_idx, s_el[v_el]]))
                sc_mean = float(np.nanmean(score_active[k_idx, s_el]))
                cands.append({
                    'k_idx': k_idx,
                    'ch_0based': analysed_channels[k_idx],
                    'el_1based': int(el_1b),
                    'pp_valid': pp_val,
                    'score_mean': sc_mean,
                })
        return cands

    # 1. Analyse main channel first
    candidates = _find_candidates_on_channel(0)
    used_neighbors = False

    # 2. Neighbor fallback ONLY if main channel has no candidates
    if len(candidates) == 0 and use_neighbors and len(nb_channels) > 0:
        used_neighbors = True
        for k_nb in range(1, n_analysed):
            candidates.extend(_find_candidates_on_channel(k_nb))

    chosen_el = None
    chosen_ch = None
    chosen_k = None
    from_neighbor = False

    if len(candidates) > 0:
        if criterion == 'score':
            candidates.sort(key=lambda c: (-c['score_mean'], -c['pp_valid']))
        else:
            candidates.sort(key=lambda c: (-c['pp_valid'], -c['score_mean']))
        winner = candidates[0]
        chosen_el = winner['el_1based']
        chosen_ch = winner['ch_0based']
        chosen_k = winner['k_idx']
        from_neighbor = not is_main_list[chosen_k]

    grid_data = {
        'role_name': role_name,
        'current_ma': current_ma,
        'active_electrodes_1based': list(active_electrodes_1based),
        'n_giri': n_giri_det,
        't_win_ms': t_win_ms,
        'analysed_channels': analysed_channels,
        'is_main': is_main_list,
        'epochs': epochs,
        'valid_mask': valid_mask,
        'el_1based_of': el_1based_of,
        'giro_of': giro_of,
        'pred': pred,
        'score_rf': score_rf,
        'score_sv': score_sv,
        'pp': pp,
        'candidates': candidates,
        'chosen_el': chosen_el,
        'chosen_ch': chosen_ch,
        'chosen_k': chosen_k,
        'from_neighbor': from_neighbor,
        'used_neighbors': used_neighbors,
        'first_skipped': first_skipped,
        'stim_idx': stim_idx,
        'fs': fs,
        'emg_raw_16ch': emg_raw_16ch,
    }

    if chosen_el is not None:
        nb_tag = f" [Neighbor Ch{chosen_ch + 1}]" if from_neighbor else f" on Ch{chosen_ch + 1}"
        msg = f"{role_name}: LOCKED El.{chosen_el} at {current_ma:.1f} mA{nb_tag}."
    else:
        msg = f"{role_name}: No valid response at {current_ma:.1f} mA."

    return {
        'found': chosen_el is not None,
        'chosen_el': chosen_el,
        'chosen_ch': chosen_ch,
        'from_neighbor': from_neighbor,
        'current_ma': current_ma,
        'used_neighbors': used_neighbors,
        'message': msg,
        'grid_data': grid_data,
    }


def run_stim_calibration(emg_raw_16ch: np.ndarray, ecr_ch: int, fcr_ch: int, fs: float = 2000.0):
    """Backward-compatible wrapper if called on a single static recording."""
    res_rad = evaluate_stim_round(
        emg_raw_16ch=emg_raw_16ch,
        target_ch=ecr_ch,
        other_main_ch=fcr_ch,
        active_electrodes_1based=[1, 2, 3, 4],
        current_ma=5.0,
        n_giri=1,
        fs=fs,
        role_name="Extensor (Radial)",
    )
    res_med = evaluate_stim_round(
        emg_raw_16ch=emg_raw_16ch,
        target_ch=fcr_ch,
        other_main_ch=ecr_ch,
        active_electrodes_1based=[5, 6, 7, 8],
        current_ma=5.0,
        n_giri=1,
        fs=fs,
        role_name="Flexor (Median)",
    )
    rad_el = (res_rad['chosen_el'] - 1) if res_rad['chosen_el'] is not None else 0
    med_el = (res_med['chosen_el'] - 1) if res_med['chosen_el'] is not None else 4
    return rad_el, med_el, {'radial': res_rad, 'median': res_med}


# Alias di compatibilità
select_stim_electrodes = run_stim_calibration