Module overview:
1. calibration_engine.py (algorithmic core)
	Contains all signal processing, envelope detection, 	thresholding, and channel selection logic:
2.calibration_worker.py: handles background data collection and 	algoithhmic execution using QThreat to prevent UI 	freezing. It manages execution for either 'read' or 	'stim' calibration phases.
3. GUI_EAST_CALIBRATION.py (Main User Interface)


Calibration Workflow
Protocol Configuration: The operator inputs a movement sequence vector (e.g., 1, 1, 0, 0, 1, 1, 0, 0).

Phase 1 - EMG Read Calibration (optional): The patient performs movements while raw 16-channel EMG is recorded. A successful calibration fills the ECR and FCR channel fields automatically.

⚠️ Mismatch Detected: Displays a warning pop-up, resets channel indices, and requires a re-trial.

✅ Match Verified: Assigns optimal ECR and FCR recording channels.

Phase 2 - Stimulation Calibration: The operator can enter distinct ECR and FCR channels manually, without running Phase 1, then set pulse parameters and execute the sweep to identify target stimulation electrodes for radial and median nerve activation. Each round automatically enables the first stimulation window and calculates its ON duration from the active electrodes, turns, and ISI.

The flowchart for the read and stimulation calibration protocol is in
FLOWCHART_CALIBRAZIONE.md.