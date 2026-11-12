import numpy as np
from scipy import signal


class MovementMismatchError(Exception):
    """Exception raised when detected activations do not match the protocol vector."""
    pass


def parse_protocol_string(protocol_str: str) -> np.ndarray:
    """Converts a string like '1,1,0,0,1,1,0,0' into a numpy array of integers."""
    tokens = [t.strip() for t in protocol_str.replace(';', ',').split(',') if t.strip()]
    return np.array([int(t) for t in tokens], dtype=int)


def _to_scalar_float(val, default=2000.0) -> float:
    """Safely converts numpy arrays/scalars or numeric inputs to a standard Python float."""
    if val is None:
        return float(default)
    if isinstance(val, (np.ndarray, list)):
        arr = np.squeeze(val)
        return float(arr.item() if arr.ndim == 0 else arr[0])
    return float(val)


def detect_bursts(env_mean: np.ndarray, threshold: float, fs: float, min_gap_ms: float = 200, min_dur_ms: float = 150):
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

    # Merge bursts with gaps shorter than min_gap
    merged_onsets, merged_offsets = [onsets[0]], []
    for i in range(len(onsets) - 1):
        if onsets[i + 1] - offsets[i] < min_gap_samples:
            continue
        else:
            merged_offsets.append(offsets[i])
            merged_onsets.append(onsets[i + 1])
    merged_offsets.append(offsets[-1])

    # Filter bursts shorter than min_dur
    final_onsets, final_offsets = [], []
    for on, off in zip(merged_onsets, merged_offsets):
        if (off - on) >= min_dur_samples:
            final_onsets.append(on)
            final_offsets.append(off)

    return np.array(final_onsets, dtype=int), np.array(final_offsets, dtype=int)


def run_read_calibration(emg_raw_16ch: np.ndarray, protocol: np.ndarray, fs: float = 2000.0):
    """
    Executes EMG Read Calibration:
    1. Computes 16-channel RMS envelope & mean envelope.
    2. Detects activation bursts and threshold.
    3. Verifies match against protocol vector.
    4. Selects highest energy channels for ECR (extension) and FCR (flexion).
    """
    fs = _to_scalar_float(fs)
    num_channels, n_samples = emg_raw_16ch.shape
    smooth_n = max(1, int(np.round(200.0 * fs / 1000.0)))
    
    # Bandpass filter 20-450 Hz
    nyq = fs / 2.0
    low = max(0.01, min(20.0 / nyq, 0.99))
    high = max(low + 0.01, min(450.0 / nyq, 0.99))
    
    b, a = signal.butter(4, [low, high], btype='bandpass')
    emg_filt = signal.filtfilt(b, a, emg_raw_16ch, axis=1)

    kernel = np.ones(smooth_n) / smooth_n
    env_all = np.zeros_like(emg_filt)
    for ch in range(num_channels):
        env_all[ch, :] = np.sqrt(np.convolve(emg_filt[ch, :] ** 2, kernel, mode='same'))

    env_mean = np.mean(env_all, axis=0)
    threshold = 1.5 * float(np.percentile(env_mean, 75))

    onsets, offsets = detect_bursts(env_mean, threshold, fs)
    detected_count = len(onsets)
    expected_count = len(protocol)

    if detected_count != expected_count:
        raise MovementMismatchError(
            f"Detected {detected_count} movement activations, but protocol expected {expected_count}."
        )

    ch_energies_ecr = np.zeros(num_channels)
    ch_energies_fcr = np.zeros(num_channels)

    for i, (on, off) in enumerate(zip(onsets, offsets)):
        trial_type = protocol[i] if i < len(protocol) else 0
        burst_energy = np.sum(env_all[:, on:off] ** 2, axis=1)
        if trial_type == 1:
            ch_energies_ecr += burst_energy
        else:
            ch_energies_fcr += burst_energy

    ecr_ch = int(np.argmax(ch_energies_ecr))
    fcr_ch = int(np.argmax(ch_energies_fcr))

    if ecr_ch == fcr_ch:
        sorted_fcr = np.argsort(ch_energies_fcr)[::-1]
        fcr_ch = int(sorted_fcr[1] if sorted_fcr[0] == ecr_ch else sorted_fcr[0])

    aux_plot_data = {
        'time': np.linspace(0, n_samples / fs, n_samples),
        'env_mean': env_mean,
        'threshold': threshold,
        'onsets': onsets / fs,
        'offsets': offsets / fs,
        'protocol': protocol,
        'ch_energies_ecr': ch_energies_ecr,
        'ch_energies_fcr': ch_energies_fcr,
        'ecr_ch': ecr_ch,
        'fcr_ch': fcr_ch
    }

    return ecr_ch, fcr_ch, aux_plot_data


# Alias di compatibilità
select_read_channels = run_read_calibration


def run_stim_calibration(emg_raw_16ch: np.ndarray, ecr_ch: int, fcr_ch: int, fs: float = 2000.0):
    """
    Executes Stimulation Calibration:
    1. Evaluates evoked responses across 8 stimulation electrodes for calibrated ECR and FCR channels.
    2. Identifies best electrode per muscle group.
    """
    fs = _to_scalar_float(fs)
    num_stim_electrodes = 8
    ecr_responses = np.zeros(num_stim_electrodes)
    fcr_responses = np.zeros(num_stim_electrodes)

    window_len = emg_raw_16ch.shape[1] // num_stim_electrodes
    for el in range(num_stim_electrodes):
        seg = emg_raw_16ch[:, el * window_len : (el + 1) * window_len]
        ecr_responses[el] = np.sqrt(np.mean(seg[ecr_ch, :] ** 2)) if seg.shape[1] > 0 else 0
        fcr_responses[el] = np.sqrt(np.mean(seg[fcr_ch, :] ** 2)) if seg.shape[1] > 0 else 0

    radial_el = int(np.argmax(ecr_responses))
    median_el = int(np.argmax(fcr_responses))

    if radial_el == median_el:
        sorted_med = np.argsort(fcr_responses)[::-1]
        median_el = int(sorted_med[1] if sorted_med[0] == radial_el else sorted_med[0])

    aux_plot_data = {
        'electrodes': np.arange(1, num_stim_electrodes + 1),
        'ecr_responses': ecr_responses,
        'fcr_responses': fcr_responses,
        'radial_el': radial_el,
        'median_el': median_el
    }

    return radial_el, median_el, aux_plot_data


# Alias di compatibilità
select_stim_electrodes = run_stim_calibration