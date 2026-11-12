from PyQt5 import QtWidgets, uic, QtGui, QtCore
from PyQt5.QtWidgets import QMessageBox
import PyQt5
import sys

import os
import serial
import numpy as np
import operator
import struct
import time
from scipy import signal
from scipy.io import savemat, loadmat
from statistics import mean
import serial.tools.list_ports
from datetime import date, datetime

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import pyqtgraph as pg

import resources_EAST  # Resource file compilato da .qrc

from calibration_worker import CalibrationWorker
from calibration_engine import (
    MovementMismatchError, 
    parse_protocol_string, 
    run_read_calibration, 
    run_stim_calibration
)

# ── BRACELET CONFIGURATION ──────────────────────────────────────────
NUM_BRACELET_CH = 16          
PLOT_COLS       = 4           
OFFSET_BRACELET_CH = [0.0] * 16

ENV_SMOOTH_MS = 200      
ENV_PERCENTILE = 75      
ONSET_THRESHOLD = 1.5    

try:
    os.chdir(sys._MEIPASS)
except Exception:
    pass


def compute_rms_envelope(emg_raw_16ch: np.ndarray, fs: float, smooth_ms: float = 200.0) -> tuple:
    """Calcola il segnale medio grezzo e l'inviluppo RMS mediato sui 16 canali."""
    num_channels, n_samples = emg_raw_16ch.shape
    smooth_n = max(1, int(round(smooth_ms * fs / 1000.0)))
    
    kernel = np.ones(smooth_n) / smooth_n
    env_all = np.zeros_like(emg_raw_16ch)
    
    for ch in range(num_channels):
        squared_signal = emg_raw_16ch[ch, :] ** 2
        env_all[ch, :] = np.sqrt(np.convolve(squared_signal, kernel, mode='same'))
    
    mean_raw_emg = np.mean(emg_raw_16ch, axis=0)
    mean_rms_env = np.mean(env_all, axis=0)
    
    return mean_raw_emg, mean_rms_env


class CalibrationEMGWindow(QtWidgets.QWidget):
    """Finestra fluttuante per lo streaming live del segnale medio e dell'inviluppo RMS."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EMG Calibration - Real-Time Signals")
        self.setWindowIcon(QtGui.QIcon(':/icons/Icons/lightning-bolt.png'))
        self.resize(900, 500)
        
        layout = QtWidgets.QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.setLabel('left', 'Amplitude', units='mV')
        self.plot_widget.addLegend()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        layout.addWidget(self.plot_widget)
        
        pen_raw = pg.mkPen(color=(180, 180, 180, 150), width=1)       
        pen_env = pg.mkPen(color=(255, 102, 0), width=3.0)            
        
        self.curve_raw = self.plot_widget.plot([], [], pen=pen_raw, name="Mean Raw EMG (16 Ch)")
        self.curve_env = self.plot_widget.plot([], [], pen=pen_env, name="Mean RMS Envelope")

    def update_data(self, time_vec: np.ndarray, raw_mean: np.ndarray, env_mean: np.ndarray):
        """Aggiorna le sole due curve in tempo reale a elevate prestazioni."""
        self.curve_raw.setData(time_vec, raw_mean)
        self.curve_env.setData(time_vec, env_mean)

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        
        ui_file = os.path.join(os.path.dirname(__file__), 'GUI_EAST_Qt_v01_names.ui')
        uic.loadUi(ui_file, self)
        self.setWindowIcon(QtGui.QIcon(':/icons/Icons/lightning-bolt.png'))
        
        # ── CONFIGURAZIONE SCHERMATA MULTIPAGINA ──────────────────────────────
        self.main_standard_widget = self.centralwidget
        self.stackedWidget = QtWidgets.QStackedWidget(self)
        self.stackedWidget.addWidget(self.main_standard_widget)
        
        self.calibration_page = QtWidgets.QWidget()
        self.setup_calibration_page_ui()
        self.stackedWidget.addWidget(self.calibration_page)

        self.stimulation_calibration_page = QtWidgets.QWidget()
        self.setup_stimulation_calibration_page_ui()
        self.stackedWidget.addWidget(self.stimulation_calibration_page)
        
        self.setCentralWidget(self.stackedWidget)

        # ── VARIABILI DI STATO E CALIBRAZIONE ──────────────────────────────────
        self.fileName = ''
        self.saveDirectory = os.getcwd()
        self.stop = 1

        self.calibrated_ecr = None
        self.calibrated_fcr = None
        self.calibrated_radial_el = None
        self.calibrated_median_el = None
        
        self.calib_plot_window = None  
        self.calib_worker = None
        self.calib_emg_chunks = []
        self.calib_mode_active = None  # 'read' or 'stim'

        # Porte COM
        ports = serial.tools.list_ports.comports()
        listPorts = [p.device for p in ports]
        self.comBox.setEditable(True)
        if listPorts:
            self.comBox.addItems(listPorts)
            self.comBox.setCurrentIndex(0)
        else:
            self.statusbar.showMessage('No COM ports found')

        # ---------- CONNESSIONI GUI STANDARD ----------
        self.saveDirectoryButton.clicked.connect(self.onSaveDirectoryButtonClicked)
        self.connectButton.clicked.connect(self.onConnectButtonClicked)
        self.disconnectButton.clicked.connect(self.onDisconnectButtonClicked)
        self.startStimButton.clicked.connect(self.onStartStimButtonClicked)
        self.stopStimButton.clicked.connect(self.onStopStimButtonClicked)
        self.startEmgButton.clicked.connect(self.onStartEmgButtonClicked)
        self.stopEmgButton.clicked.connect(self.onStopEmgButtonClicked)
        self.analyzeButton.clicked.connect(self.onAnalyzeButtonClicked)
        self.subjectBox.valueChanged.connect(self.setFileName)
        self.conditionBox.currentIndexChanged.connect(self.setFileName)
        self.trialBox.valueChanged.connect(self.setFileName)
        self.sideBox.currentIndexChanged.connect(self.setFileName)

        self.setup_calibration_radio_button()
        self.show()

    def setup_calibration_radio_button(self):
        if hasattr(self, 'manualButton') and self.manualButton.parentWidget():
            mode_layout = self.manualButton.parentWidget().layout()
            if mode_layout is not None:
                self.lectureCalibrationButton = QtWidgets.QRadioButton(
                    "Lecture Calibration", self.manualButton.parentWidget()
                )
                self.stimulationCalibrationButton = QtWidgets.QRadioButton(
                    "Stimulation Calibration", self.manualButton.parentWidget()
                )
                mode_layout.insertWidget(0, self.stimulationCalibrationButton)
                mode_layout.insertWidget(0, self.lectureCalibrationButton)
                
                if hasattr(self, 'workingModeGroup'):
                    self.workingModeGroup.addButton(self.lectureCalibrationButton)
                    self.workingModeGroup.addButton(self.stimulationCalibrationButton)

                self.lectureCalibrationButton.toggled.connect(self.on_mode_toggled)
                self.stimulationCalibrationButton.toggled.connect(self.on_mode_toggled)
                self.manualButton.toggled.connect(self.on_mode_toggled)
                if hasattr(self, 'oopButton'): self.oopButton.toggled.connect(self.on_mode_toggled)
                if hasattr(self, 'satsButton'): self.satsButton.toggled.connect(self.on_mode_toggled)
                if hasattr(self, 'conButton'): self.conButton.toggled.connect(self.on_mode_toggled)

    def on_mode_toggled(self):
        if hasattr(self, 'lectureCalibrationButton') and self.lectureCalibrationButton.isChecked():
            self.stackedWidget.setCurrentWidget(self.calibration_page)
        elif hasattr(self, 'stimulationCalibrationButton') and self.stimulationCalibrationButton.isChecked():
            self.stackedWidget.setCurrentWidget(self.stimulation_calibration_page)
        else:
            self.stackedWidget.setCurrentWidget(self.main_standard_widget)

    def copy_spin_box(self, source, parent, object_name):
        if isinstance(source, QtWidgets.QDoubleSpinBox):
            copied = QtWidgets.QDoubleSpinBox(parent)
            copied.setDecimals(source.decimals())
        else:
            copied = QtWidgets.QSpinBox(parent)

        copied.setObjectName(object_name)
        copied.setRange(source.minimum(), source.maximum())
        copied.setSingleStep(source.singleStep())
        copied.setPrefix(source.prefix())
        copied.setSuffix(source.suffix())
        copied.setValue(source.value())
        return copied

    def setup_stimulation_calibration_page_ui(self):
        self.stimulation_calibration_page.setObjectName("stimulationCalibrationPage")
        self.stimulation_calibration_page.setStyleSheet("""
            QWidget#stimulationCalibrationPage {
                background-color: #f3f6fb;
            }
            QWidget#stimulationCalibrationPage QGroupBox {
                background-color: #ffffff;
                border: 1px solid #d5deeb;
                border-radius: 8px;
                margin-top: 10px;
                padding: 10px;
                font-weight: bold;
                color: #23395d;
            }
            QWidget#stimulationCalibrationPage QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QWidget#stimulationCalibrationPage QSpinBox,
            QWidget#stimulationCalibrationPage QDoubleSpinBox,
            QWidget#stimulationCalibrationPage QTextEdit {
                min-height: 26px;
                border: 1px solid #c8d3e2;
                border-radius: 4px;
                padding: 2px 6px;
                background-color: #ffffff;
            }
            QWidget#stimulationCalibrationPage QPushButton {
                min-height: 34px;
                border: none;
                border-radius: 5px;
                padding: 4px 12px;
                background-color: #e4ebf5;
                color: #23395d;
                font-weight: bold;
            }
            QWidget#stimulationCalibrationPage QPushButton:hover {
                background-color: #d3deed;
            }
            QWidget#stimulationCalibrationPage QPushButton#startStimulationCalibrationButton {
                background-color: #198754;
                color: #ffffff;
            }
            QWidget#stimulationCalibrationPage QPushButton#startStimulationCalibrationButton:hover {
                background-color: #157347;
            }
            QWidget#stimulationCalibrationPage QPushButton#stopStimulationCalibrationButton {
                background-color: #dc3545;
                color: #ffffff;
            }
            QWidget#stimulationCalibrationPage QPushButton#stopStimulationCalibrationButton:hover {
                background-color: #bb2d3b;
            }
        """)

        layout = QtWidgets.QVBoxLayout(self.stimulation_calibration_page)
        layout.setContentsMargins(18, 14, 18, 18)
        layout.setSpacing(12)

        header = QtWidgets.QHBoxLayout()
        header.setSpacing(12)
        back_button = QtWidgets.QPushButton("← Return to Main GUI")
        back_button.clicked.connect(self.exit_calibration_mode)
        header.addWidget(back_button)

        title = QtWidgets.QLabel("STIMULATION CALIBRATION")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #23395d;")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(12)
        layout.addLayout(controls)

        left_column = QtWidgets.QVBoxLayout()
        left_column.setSpacing(12)
        controls.addLayout(left_column, stretch=3)

        sequence_box = QtWidgets.QGroupBox("Trial sequence")
        sequence_box.setToolTip(
            "Le caselle 1-8 impostano rispettivamente i bit 0-7 del parametro trialSeq."
        )
        sequence_layout = QtWidgets.QHBoxLayout(sequence_box)
        self.stimulationTrialSequenceBoxes = []
        for index in range(8):
            checkbox = QtWidgets.QCheckBox(str(index + 1), sequence_box)
            checkbox.setObjectName(f"stimulationSeq{index + 1}CheckBox")
            checkbox.setChecked(getattr(self, f"seq{index + 1}CheckBox").isChecked())
            checkbox.setToolTip(f"Imposta il bit {index} di trialSeq.")
            sequence_layout.addWidget(checkbox)
            self.stimulationTrialSequenceBoxes.append(checkbox)
        left_column.addWidget(sequence_box)

        stim_parameters_box = QtWidgets.QGroupBox("Stimulation parameters")
        stim_parameters_layout = QtWidgets.QVBoxLayout(stim_parameters_box)
        pulse_layout = QtWidgets.QFormLayout()
        self.stimulationCalibrationPulseWidth = self.copy_spin_box(
            self.pulseWidthBox, stim_parameters_box, "stimulationCalibrationPulseWidth"
        )
        self.stimulationCalibrationIsi = self.copy_spin_box(
            self.frequencyBox, stim_parameters_box, "stimulationCalibrationIsi"
        )
        pulse_layout.addRow("Pulse width (µs)", self.stimulationCalibrationPulseWidth)
        pulse_layout.addRow("ISI (s)", self.stimulationCalibrationIsi)
        stim_parameters_layout.addLayout(pulse_layout)

        phase_box = QtWidgets.QGroupBox("Stimulation phase")
        phase_layout = QtWidgets.QHBoxLayout(phase_box)
        self.stimulationCalibrationPhaseButtons = {}
        for name, text in (
            ("biphasicButton", "Biphasic"),
            ("positiveButton", "Positive"),
            ("negativeButton", "Negative"),
        ):
            button = QtWidgets.QRadioButton(text, phase_box)
            button.setObjectName(f"stimulationCalibration{name}")
            button.setChecked(getattr(self, name).isChecked())
            phase_layout.addWidget(button)
            self.stimulationCalibrationPhaseButtons[name] = button
        stim_parameters_layout.addWidget(phase_box)

        amplitudes_box = QtWidgets.QGroupBox("Stimulation channels (mA)")
        amplitudes_layout = QtWidgets.QGridLayout(amplitudes_box)
        self.stimulationCalibrationChannels = []
        for index in range(8):
            row, column = divmod(index, 2)
            channel_label = QtWidgets.QLabel(f"Channel {index + 1} (mA)")
            channel_input = self.copy_spin_box(
                getattr(self, f"stimCh{index + 1}"),
                amplitudes_box,
                f"stimulationCalibrationChannel{index + 1}",
            )
            amplitudes_layout.addWidget(channel_label, row, column * 2)
            amplitudes_layout.addWidget(channel_input, row, column * 2 + 1)
            self.stimulationCalibrationChannels.append(channel_input)
        stim_parameters_layout.addWidget(amplitudes_box)
        left_column.addWidget(stim_parameters_box)

        sats_box = QtWidgets.QGroupBox("SATS parameters")
        sats_layout = QtWidgets.QFormLayout(sats_box)
        self.stimulationCalibrationTimeOn = self.copy_spin_box(
            self.timeON, sats_box, "stimulationCalibrationTimeOn"
        )
        self.stimulationCalibrationTimeOff = self.copy_spin_box(
            self.timeOFF, sats_box, "stimulationCalibrationTimeOff"
        )
        self.stimulationCalibrationEcrChannel = self.copy_spin_box(
            self.ecrChannel, sats_box, "stimulationCalibrationEcrChannel"
        )
        self.stimulationCalibrationFcrChannel = self.copy_spin_box(
            self.fcrChannel, sats_box, "stimulationCalibrationFcrChannel"
        )
        sats_layout.addRow("Time stimulation ON", self.stimulationCalibrationTimeOn)
        sats_layout.addRow("Time stimulation OFF", self.stimulationCalibrationTimeOff)
        sats_layout.addRow("ECR channel (1-16)", self.stimulationCalibrationEcrChannel)
        sats_layout.addRow("FCR channel (1-16)", self.stimulationCalibrationFcrChannel)
        controls.addWidget(sats_box, stretch=1)

        buttons = QtWidgets.QHBoxLayout()
        buttons.setSpacing(12)
        self.startStimulationCalibrationButton = QtWidgets.QPushButton("Start Stimulation")
        self.startStimulationCalibrationButton.setObjectName("startStimulationCalibrationButton")
        self.startStimulationCalibrationButton.clicked.connect(self.onStartStimButtonClicked)
        self.stopStimulationCalibrationButton = QtWidgets.QPushButton("Stop")
        self.stopStimulationCalibrationButton.setObjectName("stopStimulationCalibrationButton")
        self.stopStimulationCalibrationButton.clicked.connect(self.onStopStimButtonClicked)
        buttons.addWidget(self.startStimulationCalibrationButton)
        buttons.addWidget(self.stopStimulationCalibrationButton)
        layout.addLayout(buttons)

        recording_box = QtWidgets.QGroupBox("Recording and data")
        recording_layout = QtWidgets.QFormLayout(recording_box)
        self.stimulationCalibrationRecordingTime = self.copy_spin_box(
            self.recordingTimeBox, recording_box, "stimulationCalibrationRecordingTime"
        )
        self.stimulationCalibrationFileName = QtWidgets.QTextEdit(recording_box)
        self.stimulationCalibrationFileName.setObjectName("stimulationCalibrationFileName")
        self.stimulationCalibrationFileName.setMaximumHeight(40)
        self.stimulationCalibrationFileName.setPlainText(self.nameEdit.toPlainText())

        self.stimulationCalibrationSaveDirectoryButton = QtWidgets.QPushButton(
            "Choose directory", recording_box
        )
        self.stimulationCalibrationSaveDirectoryButton.clicked.connect(
            self.onSaveDirectoryButtonClicked
        )
        self.stimulationCalibrationSaveCheckBox = QtWidgets.QCheckBox("Save Data", recording_box)
        self.stimulationCalibrationSaveCheckBox.setChecked(self.saveCheckBox.isChecked())

        recording_layout.addRow("Recording time (s)", self.stimulationCalibrationRecordingTime)
        recording_layout.addRow("File name", self.stimulationCalibrationFileName)
        recording_layout.addRow(
            self.stimulationCalibrationSaveDirectoryButton,
            self.stimulationCalibrationSaveCheckBox,
        )
        layout.addWidget(recording_box)

    def setup_calibration_page_ui(self):
        """Costruisce la UI di calibrazione con i pulsanti Start e Stop."""
        main_layout = QtWidgets.QVBoxLayout(self.calibration_page)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        top_panel = QtWidgets.QWidget()
        top_layout = QtWidgets.QHBoxLayout(top_panel)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        # Colonna 1: Protocollo
        left_ctrl = QtWidgets.QVBoxLayout()
        self.btn_back_to_standard = QtWidgets.QPushButton("← Return to Main GUI")
        self.btn_back_to_standard.setStyleSheet("font-weight: bold; padding: 6px; background-color: #e0e0e0;")
        self.btn_back_to_standard.clicked.connect(self.exit_calibration_mode)
        left_ctrl.addWidget(self.btn_back_to_standard)

        title_label = QtWidgets.QLabel("CALIBRATION MODE")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #0d6efd;")
        left_ctrl.addWidget(title_label)

        proto_box = QtWidgets.QGroupBox("1. Protocol Configuration")
        proto_layout = QtWidgets.QVBoxLayout(proto_box)
        self.protocolInputLabel = QtWidgets.QLabel("Protocol Vector (e.g. 1,1,0,0,1,1,0,0):")
        self.protocolLineEdit = QtWidgets.QLineEdit("1, 1, 0, 0, 1, 1, 0, 0")
        proto_layout.addWidget(self.protocolInputLabel)
        proto_layout.addWidget(self.protocolLineEdit)
        left_ctrl.addWidget(proto_box)
        
        top_layout.addLayout(left_ctrl)

        # Colonna 2: Controlli Start/Stop per le 2 Fasi
        mid_ctrl = QtWidgets.QVBoxLayout()
        
        # Fase 2: Read Calibration
        read_box = QtWidgets.QGroupBox("2. EMG Read Calibration")
        read_layout = QtWidgets.QVBoxLayout(read_box)
        
        btn_read_layout = QtWidgets.QHBoxLayout()
        self.startReadCalibBtn = QtWidgets.QPushButton("Start EMG Recording")
        self.startReadCalibBtn.setStyleSheet("background-color: #0d6efd; color: white; font-weight: bold; padding: 6px;")
        self.startReadCalibBtn.clicked.connect(self.start_read_calibration)
        
        self.stopReadCalibBtn = QtWidgets.QPushButton("Stop EMG Recording")
        self.stopReadCalibBtn.setEnabled(False)
        self.stopReadCalibBtn.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold; padding: 6px;")
        self.stopReadCalibBtn.clicked.connect(self.stop_read_calibration)
        
        btn_read_layout.addWidget(self.startReadCalibBtn)
        btn_read_layout.addWidget(self.stopReadCalibBtn)
        
        self.readStatusLabel = QtWidgets.QLabel("Read Status: Waiting")
        read_layout.addLayout(btn_read_layout)
        read_layout.addWidget(self.readStatusLabel)
        mid_ctrl.addWidget(read_box)

        # Fase 3: Stimulation Calibration
        stim_box = QtWidgets.QGroupBox("3. Stimulation Calibration")
        stim_layout = QtWidgets.QVBoxLayout(stim_box)
        
        btn_stim_layout = QtWidgets.QHBoxLayout()
        self.startStimCalibBtn = QtWidgets.QPushButton("Start Stimulation")
        self.startStimCalibBtn.setEnabled(False)
        self.startStimCalibBtn.setStyleSheet("background-color: #198754; color: white; font-weight: bold; padding: 6px;")
        self.startStimCalibBtn.clicked.connect(self.start_stim_calibration)

        self.stopStimCalibBtn = QtWidgets.QPushButton("Stop Stimulation")
        self.stopStimCalibBtn.setEnabled(False)
        self.stopStimCalibBtn.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold; padding: 6px;")
        self.stopStimCalibBtn.clicked.connect(self.stop_stim_calibration)

        btn_stim_layout.addWidget(self.startStimCalibBtn)
        btn_stim_layout.addWidget(self.stopStimCalibBtn)

        self.stimStatusLabel = QtWidgets.QLabel("Stimulation Status: Disabled (Complete Step 1)")
        stim_layout.addLayout(btn_stim_layout)
        stim_layout.addWidget(self.stimStatusLabel)
        mid_ctrl.addWidget(stim_box)

        top_layout.addLayout(mid_ctrl)

        # Colonna 3: Risultati
        res_box = QtWidgets.QGroupBox("Calibration Results")
        res_layout = QtWidgets.QVBoxLayout(res_box)
        self.lbl_ecr_res = QtWidgets.QLabel("ECR Channel: Unassigned")
        self.lbl_fcr_res = QtWidgets.QLabel("FCR Channel: Unassigned")
        self.lbl_rad_res = QtWidgets.QLabel("Radial Electrode: Unassigned")
        self.lbl_med_res = QtWidgets.QLabel("Median Electrode: Unassigned")
        res_layout.addWidget(self.lbl_ecr_res)
        res_layout.addWidget(self.lbl_fcr_res)
        res_layout.addWidget(self.lbl_rad_res)
        res_layout.addWidget(self.lbl_med_res)
        top_layout.addWidget(res_box)

        main_layout.addWidget(top_panel)

        # Grafici Ausiliari in basso
        aux_box = QtWidgets.QGroupBox("AUXILIARY CALIBRATION PLOTS")
        aux_box.setStyleSheet("font-weight: bold; color: #333;")
        aux_layout = QtWidgets.QVBoxLayout(aux_box)

        self.aux_tab_widget = QtWidgets.QTabWidget()

        self.tab_read_plot = QtWidgets.QWidget()
        tab1_layout = QtWidgets.QHBoxLayout(self.tab_read_plot)
        
        self.pw_read_envelope = pg.PlotWidget()
        self.pw_read_envelope.setBackground('w')
        self.pw_read_envelope.setTitle("Mean RMS Envelope & Detected Onsets", size='9pt')
        self.pw_read_envelope.setLabel('bottom', 'Time', units='s')
        self.pw_read_envelope.setLabel('left', 'Amplitude', units='mV')
        
        self.pw_channel_energy = pg.PlotWidget()
        self.pw_channel_energy.setBackground('w')
        self.pw_channel_energy.setTitle("Channel Activation Profiles (16 Ch)", size='9pt')
        self.pw_channel_energy.setLabel('bottom', 'Channel ID')
        self.pw_channel_energy.setLabel('left', 'Energy')

        tab1_layout.addWidget(self.pw_read_envelope, stretch=2)
        tab1_layout.addWidget(self.pw_channel_energy, stretch=1)
        self.aux_tab_widget.addTab(self.tab_read_plot, "Read Calibration Analysis")

        self.tab_stim_plot = QtWidgets.QWidget()
        tab2_layout = QtWidgets.QVBoxLayout(self.tab_stim_plot)

        self.pw_stim_response = pg.PlotWidget()
        self.pw_stim_response.setBackground('w')
        self.pw_stim_response.setTitle("Evoked Responses per Electrode (ECR vs FCR)", size='9pt')
        self.pw_stim_response.setLabel('bottom', 'Stimulation Electrode')
        self.pw_stim_response.setLabel('left', 'Response RMS')
        self.pw_stim_response.addLegend()

        tab2_layout.addWidget(self.pw_stim_response)
        self.aux_tab_widget.addTab(self.tab_stim_plot, "Stimulation Calibration Analysis")

        aux_layout.addWidget(self.aux_tab_widget)
        main_layout.addWidget(aux_box, stretch=1)

    def exit_calibration_mode(self):
        if hasattr(self, 'manualButton'):
            self.manualButton.setChecked(True)

    def send_start_EMG_command(self):
        """Invia il pacchetto seriale di configurazione a 25 byte al firmware STM32."""
        commnadSize = 25
        stimFreqDataSet = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100])
        stimDurDataSet = np.array([100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 750, 800, 850, 900, 950, 1000])

        confString = [commnadSize]

        b2 = 0
        if self.params['fs'] == 16000: b2 |= int('00110000', 2)
        elif self.params['fs'] == 2000: b2 |= int('00000000', 2)
        elif self.params['fs'] == 4000: b2 |= int('00010000', 2)
        elif self.params['fs'] == 8000: b2 |= int('00100000', 2)

        if self.params['emgChannels'] == 2: b2 |= int('00000000', 2)
        elif self.params['emgChannels'] == 4: b2 |= int('00000100', 2)
        elif self.params['emgChannels'] == 8: b2 |= int('00001000', 2)
        elif self.params['emgChannels'] == 16: b2 |= int('00001100', 2)

        b2 += 1
        confString.append(b2)
        confString.append(0)

        stimFreqIndex = np.argmin(abs(stimFreqDataSet - self.params['stimFrequency'])) + 1
        stimType = 3*32 if self.params['stimPhase'] == 'Biphasic' else (1*32 if self.params['stimPhase'] == 'Positive' else 2*32)
        confString.append(stimType + stimFreqIndex)

        stimDurIndex = np.argmin(abs(stimDurDataSet - self.params['pulseWidth'])) + 1
        confString.append(stimDurIndex)

        for _ in range(8): confString.append(0)

        confString.append(self.params['timeON'])
        confString.append(self.params['timeOFF'])
        confString.append(self.params['timeRead'])
        confString.append(self.params['timeStim'])
        confString.append(self.params['trialSeq'])
        confString.append(0)
        confString.append(self.params['tremorThreshold'])
        confString.append(self.params['gain1'])
        confString.append(self.params['gain2'])
        confString.append(self.params['timeRMS'])
        confString.append((self.params['ecrChannel'] << 4) | self.params['fcrChannel'])

        confString.append(self.CRC8(confString, commnadSize - 1))
        sendConfstring = [int(x) for x in confString]
        
        self.serialPort.write(struct.pack("{0}B".format(len(sendConfstring)), *sendConfstring))

    # ── LOGICA CALIBRAZIONE DI LETTURA (EMG READ) ───────────────────────────
    def start_read_calibration(self):
        """Avvia la registrazione EMG a 16 canali e apre la finestra con il grafico in tempo reale."""
        if not hasattr(self, 'serialPort') or self.serialPort is None or not self.serialPort.is_open:
            QMessageBox.critical(self, "Error", "Hardware EAST non connesso! Connettere la porta COM prima di proseguire.")
            return

        self.getParameters()
        self.calib_emg_chunks = []
        self.calib_mode_active = 'read'

        self.startReadCalibBtn.setEnabled(False)
        self.stopReadCalibBtn.setEnabled(True)
        self.readStatusLabel.setText("Read Status: Recording in progress... Press STOP when finished.")

        if self.calib_plot_window is None:
            self.calib_plot_window = CalibrationEMGWindow()
        self.calib_plot_window.show()

        self.calib_worker = CalibrationWorker(ui=self)
        self.calib_worker.signal_chunk.connect(self.on_calibration_chunk_received)
        self.calib_worker.signal_error.connect(self.on_calibration_error)
        self.calib_worker.start()

    def on_calibration_chunk_received(self, chunk_16ch: np.ndarray):
        """Riceve ciascun blocco dal worker e aggiorna il grafico senza overhead di calcolo soglia."""
        self.calib_emg_chunks.append(chunk_16ch)

        if self.calib_mode_active == 'read' and self.calib_plot_window is not None:
            full_emg = np.hstack(self.calib_emg_chunks)
            fs = self.params.get('fs', 2000)
            
            raw_mean, env_mean = compute_rms_envelope(full_emg, fs)
            time_vec = np.linspace(0, full_emg.shape[1] / fs, full_emg.shape[1])
            
            self.calib_plot_window.update_data(time_vec, raw_mean, env_mean)

    def stop_read_calibration(self):
        """Arresta la registrazione ed invia i dati accumulati a run_read_calibration()."""
        if self.calib_worker is not None:
            self.calib_worker.stop()
            self.calib_worker = None

        self.stopReadCalibBtn.setEnabled(False)
        self.startReadCalibBtn.setEnabled(True)
        self.readStatusLabel.setText("Read Status: Processing recorded data...")

        if len(self.calib_emg_chunks) == 0:
            QMessageBox.warning(self, "Warning", "Nessun dato registrato. Ripetere l'operazione.")
            self.readStatusLabel.setText("Read Status: Waiting")
            return

        emg_raw_16ch = np.hstack(self.calib_emg_chunks)
        fs = self.params.get('fs', 2000)
        protocol_str = self.protocolLineEdit.text()
        protocol = parse_protocol_string(protocol_str)

        try:
            ecr_ch, fcr_ch, aux_plot_data = run_read_calibration(
                emg_raw_16ch=emg_raw_16ch,
                protocol=protocol,
                fs=fs
            )
            self.on_read_calibration_success(ecr_ch, fcr_ch, aux_plot_data)
        except MovementMismatchError as e:
            self.on_read_calibration_mismatch(str(e))
        except Exception as e:
            self.on_calibration_error(str(e))

    def on_read_calibration_success(self, ecr_ch, fcr_ch, aux_plot_data):
        self.calibrated_ecr = ecr_ch
        self.calibrated_fcr = fcr_ch
        
        if hasattr(self, 'ecrChannel'): self.ecrChannel.setValue(ecr_ch + 1)
        if hasattr(self, 'fcrChannel'): self.fcrChannel.setValue(fcr_ch + 1)

        self.readStatusLabel.setText(f"Read Status: COMPLETED (ECR: Ch{ecr_ch + 1}, FCR: Ch{fcr_ch + 1})")
        self.lbl_ecr_res.setText(f"ECR Channel: Ch {ecr_ch + 1}")
        self.lbl_fcr_res.setText(f"FCR Channel: Ch {fcr_ch + 1}")

        # Aggiornamento Grafico Ausiliario 1 (Inviluppo e Onsets)
        self.pw_read_envelope.clear()
        time_vec = aux_plot_data.get('time', [])
        env_mean = aux_plot_data.get('env_mean', [])
        thresh = aux_plot_data.get('threshold', 0)

        pen_env = pg.mkPen(color=(255, 102, 0), width=2.5)
        pen_thresh = pg.mkPen(color=(220, 20, 60), width=1.5, style=QtCore.Qt.DashLine)
        
        self.pw_read_envelope.plot(time_vec, env_mean, pen=pen_env)
        self.pw_read_envelope.plot(time_vec, np.full_like(time_vec, thresh), pen=pen_thresh)

        for on, off in zip(aux_plot_data.get('onsets', []), aux_plot_data.get('offsets', [])):
            region = pg.LinearRegionItem(values=[on, off], brush=(0, 255, 0, 50), movable=False)
            self.pw_read_envelope.addItem(region)

        # Aggiornamento Grafico Ausiliario 1 (Profili Energetici Canali)
        self.pw_channel_energy.clear()
        ch_x = np.arange(1, 17)
        ecr_e = aux_plot_data.get('ch_energies_ecr', np.zeros(16))
        fcr_e = aux_plot_data.get('ch_energies_fcr', np.zeros(16))

        bg_ecr = pg.BarGraphItem(x=ch_x - 0.2, height=ecr_e, width=0.4, brush='b', name='ECR Trials')
        bg_fcr = pg.BarGraphItem(x=ch_x + 0.2, height=fcr_e, width=0.4, brush='r', name='FCR Trials')
        self.pw_channel_energy.addItem(bg_ecr)
        self.pw_channel_energy.addItem(bg_fcr)

        self.aux_tab_widget.setCurrentIndex(0)
        self.startStimCalibBtn.setEnabled(True)
        self.stimStatusLabel.setText("Stimulation Status: Ready for execution")

    def on_read_calibration_mismatch(self, warning_msg):
        self.calibrated_ecr = None
        self.calibrated_fcr = None
        self.startStimCalibBtn.setEnabled(False)
        self.startReadCalibBtn.setEnabled(True)
        self.readStatusLabel.setText("Read Status: FAILED (Please repeat)")
        self.stimStatusLabel.setText("Stimulation Status: Disabled")

        QMessageBox.warning(self, "Movement Mismatch", f"WARNING: {warning_msg}\n\nPlease repeat the recording.")

    # ── LOGICA CALIBRAZIONE STIMOLAZIONE ────────────────────────────────────
    def start_stim_calibration(self):
        """Avvia l'acquisizione delle risposte evocate per la stimolazione."""
        if not hasattr(self, 'serialPort') or self.serialPort is None or not self.serialPort.is_open:
            QMessageBox.critical(self, "Error", "Hardware EAST non connesso!")
            return

        if self.calibrated_ecr is None or self.calibrated_fcr is None:
            QMessageBox.critical(self, "Error", "Completare prima la Calibrazione di Lettura.")
            return

        self.getParameters()
        self.calib_emg_chunks = []
        self.calib_mode_active = 'stim'

        self.startStimCalibBtn.setEnabled(False)
        self.stopStimCalibBtn.setEnabled(True)
        self.stimStatusLabel.setText("Stimulation Status: Running sweep... Press STOP when finished.")

        self.calib_worker = CalibrationWorker(ui=self)
        self.calib_worker.signal_chunk.connect(self.on_calibration_chunk_received)
        self.calib_worker.signal_error.connect(self.on_calibration_error)
        self.calib_worker.start()

    def stop_stim_calibration(self):
        """Arresta lo sweep ed invia i dati accumulati a run_stim_calibration()."""
        if self.calib_worker is not None:
            self.calib_worker.stop()
            self.calib_worker = None

        self.stopStimCalibBtn.setEnabled(False)
        self.startStimCalibBtn.setEnabled(True)
        self.stimStatusLabel.setText("Stimulation Status: Processing sweep data...")

        if len(self.calib_emg_chunks) == 0:
            QMessageBox.warning(self, "Warning", "Nessun dato di stimolazione registrato.")
            self.stimStatusLabel.setText("Stimulation Status: Ready for execution")
            return

        emg_raw_16ch = np.hstack(self.calib_emg_chunks)
        fs = self.params.get('fs', 2000)

        try:
            rad_el, med_el, aux_plot_data = run_stim_calibration(
                emg_raw_16ch=emg_raw_16ch,
                ecr_ch=self.calibrated_ecr,
                fcr_ch=self.calibrated_fcr,
                fs=fs
            )
            self.on_stim_calibration_success(rad_el, med_el, aux_plot_data)
        except Exception as e:
            self.on_calibration_error(str(e))

    def on_stim_calibration_success(self, rad_el, med_el, aux_plot_data):
        self.calibrated_radial_el = rad_el
        self.calibrated_median_el = med_el

        self.stimStatusLabel.setText(f"Stimulation Status: COMPLETED (Radial: El {rad_el + 1}, Median: El {med_el + 1})")
        self.lbl_rad_res.setText(f"Radial Electrode: El {rad_el + 1}")
        self.lbl_med_res.setText(f"Median Electrode: El {med_el + 1}")

        # Aggiornamento Grafico Ausiliario 2 (Risposte Evocate)
        self.pw_stim_response.clear()
        electrodes = aux_plot_data.get('electrodes', np.arange(1, 9))
        ecr_resp = aux_plot_data.get('ecr_responses', np.zeros(8))
        fcr_resp = aux_plot_data.get('fcr_responses', np.zeros(8))

        pen_ecr = pg.mkPen(color=(0, 102, 204), width=2.5)
        pen_fcr = pg.mkPen(color=(204, 0, 0), width=2.5)

        self.pw_stim_response.plot(electrodes, ecr_resp, pen=pen_ecr, symbol='o', symbolSize=8, name="ECR Channel Response")
        self.pw_stim_response.plot(electrodes, fcr_resp, pen=pen_fcr, symbol='s', symbolSize=8, name="FCR Channel Response")

        self.aux_tab_widget.setCurrentIndex(1)

    def on_calibration_error(self, err_msg):
        if self.calib_worker is not None:
            self.calib_worker.stop()
            self.calib_worker = None

        self.startReadCalibBtn.setEnabled(True)
        self.stopReadCalibBtn.setEnabled(False)
        self.startStimCalibBtn.setEnabled(self.calibrated_ecr is not None)
        self.stopStimCalibBtn.setEnabled(False)

        QMessageBox.critical(self, "Calibration Error", f"An error occurred: {err_msg}")

    # ── METODI GUI STANDARD ──────────────────────────────────────────────────
    def setFileName(self):
        subject = str(self.subjectBox.value())
        if len(subject) < 2: subject = '0' + subject
        conditon = self.conditionBox.currentText()
        side = self.sideBox.currentText()
        trial = str(self.trialBox.value())
        if len(trial) < 2: trial = '0' + trial
        currentDate = str(date.today()).replace('-','')
        self.fileName = currentDate + '_subject_' + subject + '_cond_' + conditon + '-' + side + '_run_' + trial
        self.nameEdit.setText(self.fileName)

    def onStartStimButtonClicked(self):
        self.getParameters()
        self.extraWindow = AnotherWindow(num_ch=2)
        self.extraWindow.show()
        commandSent = self.fcn_start_stim()

    def onStopStimButtonClicked(self):
        commandSent = self.fcn_stop_stim()

    def onStartEmgButtonClicked(self):
        self.getParameters()
        self.extraWindow = AnotherWindow(num_ch=self.params['emgChannels'])
        self.extraWindow.show()
        commandSent = self.fcn_start_EMG()

    def onStopEmgButtonClicked(self):
        commandSent = self.fcn_stop_stim()

    def getParameters(self):
        self.params = {}
        if self.manualButton.isChecked(): self.params['mode'] = 'Manual'
        elif hasattr(self, 'oopButton') and self.oopButton.isChecked(): self.params['mode'] = 'Out-of-phase'
        elif hasattr(self, 'satsButton') and self.satsButton.isChecked(): self.params['mode'] = 'SATS'
        elif hasattr(self, 'conButton') and self.conButton.isChecked(): self.params['mode'] = 'CON'
        elif hasattr(self, 'lectureCalibrationButton') and self.lectureCalibrationButton.isChecked(): self.params['mode'] = 'Calibration'
        elif hasattr(self, 'stimulationCalibrationButton') and self.stimulationCalibrationButton.isChecked(): self.params['mode'] = 'CON'

        stimulation_calibration = (
            hasattr(self, 'stimulationCalibrationButton')
            and self.stimulationCalibrationButton.isChecked()
        )
        if stimulation_calibration:
            pulse_width_box = self.stimulationCalibrationPulseWidth
            frequency_box = self.stimulationCalibrationIsi
            phase_buttons = self.stimulationCalibrationPhaseButtons
            amplitude_boxes = self.stimulationCalibrationChannels
            sequence_boxes = self.stimulationTrialSequenceBoxes
        else:
            pulse_width_box = self.pulseWidthBox
            frequency_box = self.frequencyBox
            phase_buttons = {
                'biphasicButton': self.biphasicButton,
                'positiveButton': self.positiveButton,
                'negativeButton': self.negativeButton,
            }
            amplitude_boxes = [getattr(self, f"stimCh{index}") for index in range(1, 9)]
            sequence_boxes = [getattr(self, f"seq{index}CheckBox") for index in range(1, 9)]

        self.params['pulseWidth'] = pulse_width_box.value()
        self.params['stimFrequency'] = frequency_box.value()

        if phase_buttons['biphasicButton'].isChecked(): self.params['stimPhase'] = 'Biphasic'
        elif phase_buttons['positiveButton'].isChecked(): self.params['stimPhase'] = 'Positive'
        elif phase_buttons['negativeButton'].isChecked(): self.params['stimPhase'] = 'Negative'

        self.params['stimAmplitude'] = [box.value() for box in amplitude_boxes]

        trialSeq = [box.isChecked() << index for index, box in enumerate(sequence_boxes)]
        self.params['trialSeq'] = ((((((trialSeq[0] | trialSeq[1]) | trialSeq[2]) | trialSeq[3]) | trialSeq[4]) | trialSeq[5]) | trialSeq[6]) | trialSeq[7]

        if stimulation_calibration:
            time_on_box = self.stimulationCalibrationTimeOn
            time_off_box = self.stimulationCalibrationTimeOff
            ecr_channel_box = self.stimulationCalibrationEcrChannel
            fcr_channel_box = self.stimulationCalibrationFcrChannel
        else:
            time_on_box = self.timeON
            time_off_box = self.timeOFF
            ecr_channel_box = self.ecrChannel
            fcr_channel_box = self.fcrChannel

        self.params['timeON'] = time_on_box.value()
        self.params['timeOFF'] = time_off_box.value()
        self.params['timeRead'] = self.timeRead.value()
        self.params['timeStim'] = self.timeStim.value()
        self.params['tremorThreshold'] = int(self.tremorThreshold.value())
        
        self.params['ecrChannel']   = ecr_channel_box.value() - 1
        self.params['fcrChannel']   = fcr_channel_box.value() - 1
        self.params['tremorMuscle'] = self.params['ecrChannel']
        
        self.params['numPulses'] = self.numPulses.value()
        self.params['numTrials'] = self.numTrials.value()
        self.params['gain1'] = int(self.gain1.value()*10)
        self.params['gain2'] = int(self.gain2.value()*10)
        self.params['timeRMS'] = self.timeRMS.value()

        if self.fs16000radioButton.isChecked(): self.params['fs'] = 16000
        elif self.fs2000radioButton.isChecked(): self.params['fs'] = 2000
        elif self.fs4000radioButton.isChecked(): self.params['fs'] = 4000
        elif self.fs8000radioButton.isChecked(): self.params['fs'] = 8000

        if self.params.get('mode') == 'Calibration':
            self.params['emgChannels'] = 16
        else:
            if self.ch2radioButton.isChecked(): self.params['emgChannels'] = 2
            elif self.ch4radioButton.isChecked(): self.params['emgChannels'] = 4
            elif self.ch8radioButton.isChecked(): self.params['emgChannels'] = 8
            elif self.ch16radioButton.isChecked(): self.params['emgChannels'] = 16

        if stimulation_calibration:
            recording_time_box = self.stimulationCalibrationRecordingTime
            file_name_edit = self.stimulationCalibrationFileName
            save_data_checkbox = self.stimulationCalibrationSaveCheckBox
        else:
            recording_time_box = self.recordingTimeBox
            file_name_edit = self.nameEdit
            save_data_checkbox = self.saveCheckBox

        self.params['recordingTime'] = recording_time_box.value()
        self.fileName = file_name_edit.toPlainText()
        self.params['saveData'] = save_data_checkbox.isChecked()
        self.params['stimulationCalibration'] = stimulation_calibration

    def onConnectButtonClicked(self):
        connected = self.fcn_connect_hardware()
        if connected:
            self.connectButton.setStyleSheet("background-color : rgb(9, 249, 17)")
            self.disconnectButton.setStyleSheet("background-color : red")
        else:
            self.connectButton.setStyleSheet("background-color : red")

    def onDisconnectButtonClicked(self):
        connected = self.fcn_disconnect_hardware()
        if not connected:
            self.disconnectButton.setStyleSheet("background-color : #0d6efd")
            self.connectButton.setStyleSheet("background-color : #0d6efd")
        else:
            self.disconnectButton.setStyleSheet("background-color : orange")
            self.connectButton.setStyleSheet("background-color : #0d6efd")

    def onSaveDirectoryButtonClicked(self):
        self.saveDirectory = str(QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Directory to Save Data'))
        self.statusbar.showMessage(('Saving files in ' + self.saveDirectory))

    def fcn_connect_hardware(self):
        plotTime = 1
        fs = 2000
        numCh = 16
        numChTot = numCh + 1
        bufSize = fs * plotTime * 2 * numChTot
        serialCOM = str(self.comBox.currentText())

        self.serialPort = serial.Serial()
        self.serialPort.baudrate = 921600
        self.serialPort.port = serialCOM
        self.serialPort.timeout = 12
        try:
            self.serialPort.open()
            connected = 1
            self.serialPort.set_buffer_size(rx_size=bufSize)
            self.stop = 0
        except Exception:
            print('Not able to connect to EAST!')
            connected = 0
        return connected

    def fcn_disconnect_hardware(self):
        try:
            self.serialPort.close()
            connected = 0
        except Exception:
            connected = 1
        return connected

    def fcn_start_EMG(self):
        commnadSize = 25
        self.stop = 0
        stimFreqDataSet = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100])
        stimDurDataSet = np.array([100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 750, 800, 850, 900, 950, 1000])

        confString = [commnadSize, int('00000000', 2)]
        if self.params['fs'] == 16000: confString[-1] += int('00110000', 2)
        elif self.params['fs'] == 2000: confString[-1] += int('00000000', 2)
        elif self.params['fs'] == 4000: confString[-1] += int('00010000', 2)
        elif self.params['fs'] == 8000: confString[-1] += int('00100000', 2)

        if self.params['emgChannels'] == 2: confString[-1] += int('00000000', 2)
        elif self.params['emgChannels'] == 4: confString[-1] += int('00000100', 2)
        elif self.params['emgChannels'] == 8: confString[-1] += int('00001000', 2)
        elif self.params['emgChannels'] == 16: confString[-1] += int('00001100', 2)

        confString[-1] += 1
        confString.append(0)

        stimFreqIndex = np.argmin(abs(stimFreqDataSet - self.params['stimFrequency'])) + 1
        stimType = 3 * 32 if self.params['stimPhase'] == 'Biphasic' else (1 * 32 if self.params['stimPhase'] == 'Positive' else 2 * 32)
        confString.append(stimType + stimFreqIndex)

        stimDurIndex = np.argmin(abs(stimDurDataSet - self.params['pulseWidth'])) + 1
        confString.append(stimDurIndex)

        for _ in range(8): confString.append(0)

        confString.append(self.params['timeON'])
        confString.append(self.params['timeOFF'])
        confString.append(self.params['timeRead'])
        confString.append(self.params['timeStim'])
        confString.append(self.params['trialSeq'])

        confString.append(0)
        confString.append(self.params['tremorThreshold'])
        confString.append(self.params['gain1'])
        confString.append(self.params['gain2'])
        confString.append(self.params['timeRMS'])
        confString.append((self.params['ecrChannel'] << 4) | self.params['fcrChannel'])

        confString.append(self.CRC8(confString, commnadSize - 1))
        sendConfstring = [int(x) for x in confString]

        self.serialPort.write(struct.pack("{0}B".format(len(sendConfstring)), *sendConfstring))
        self.readEMG()

        commandSent = 1
        self.stop = 0
        return commandSent

    def readEMG(self, display_channels=None):
        self.stop = 0
        numChannelsTot = self.params['emgChannels'] + 1
        tRead = self.params['recordingTime']
        plotTime = 1
        offset = None

        convFactor = 0.000286 * 1000
        nSamples = 125
        bytesExpected = 2 * numChannelsTot * nSamples 
        bytesRead = 0
        currentBytesData = []
        self.emgRaw = []

        numCyc = tRead * self.params['fs'] / nSamples
        currentCyc = 0
        if numCyc == 0: numCyc = 99999

        numSamples = self.params['fs'] * plotTime
        t = np.linspace(0, plotTime, numSamples)

        order = 200
        wn = 10.0 / (self.params['fs'] / 2.0)
        fir_bLP = signal.firwin(order + 1, wn, window='hamming', pass_zero='lowpass')
        self.serialPort.reset_input_buffer()

        while currentCyc < numCyc:
            self.getParameters()
            y_plot = []
            y_add = []

            if self.stop == 1: break

            if self.serialPort.in_waiting >= bytesExpected:
                currentCyc += 1
                if self.stop == 0:
                    currentBytesData = []
                    chunk = self.serialPort.read(bytesExpected)
                    if chunk == b'': raise RuntimeError('Serial connection broken')

                    currentBytesData.append(chunk)
                    bytesRead += len(chunk)
                    currentBytesDataJoined = b''.join(currentBytesData)
                    currentData = np.array(
                        struct.unpack("{0}h".format(int(len(currentBytesDataJoined)/2)), currentBytesDataJoined), 
                        dtype=np.float64
                    ).reshape((numChannelsTot, nSamples), order="F")

                    spurious_now = int(np.argmin(np.var(currentData, axis=1)))
                    shift_back = (numChannelsTot - 1 - spurious_now) % numChannelsTot
                    if shift_back != 0: currentData = np.roll(currentData, shift_back, axis=0)

                    if offset is None:
                        spuriousIdx = int(np.argmin(np.var(currentData, axis=1)))
                        offset = [0.0] * numChannelsTot
                        emgPositions = [p for p in range(numChannelsTot) if p != spuriousIdx]
                        for pos, val in zip(emgPositions, OFFSET_BRACELET_CH):
                            if pos < len(offset):
                                offset[pos] = val

                    if type(self.emgRaw) == list: self.emgRaw = currentData
                    else: self.emgRaw = np.append(self.emgRaw, currentData, axis=1)

                    requested_channels = display_channels if display_channels is not None else list(range(self.params['emgChannels']))
                    valid_channels = [ch for ch in requested_channels if ch < currentData.shape[0]]

                    for i in valid_channels:
                        y_plot.append(currentData[i, :] * convFactor - offset[i])

                    for i in valid_channels:
                        env = signal.lfilter(fir_bLP, 1, np.abs(currentData[i, :] * convFactor))
                        y_add.append(env)
                        
                    QtWidgets.QApplication.processEvents()
                    self.extraWindow.updateSeparatePlot(x_to_plot=t, y_to_plot=y_plot, additional_x=t, additional_y=[])

        for j in range(numChannelsTot):
            self.emgRaw[j, :] = self.emgRaw[j, :] * convFactor - offset[j]

        commandSent = self.fcn_stop_stim()
        if not self.params.get('saveData', self.saveCheckBox.isChecked()):
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setWindowTitle('EAST GUI')
            msg.setText('Do you want to save the recorded data?')
            msg.setInformativeText('It will be saved in ' + self.saveDirectory)
            msg.setStandardButtons(QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard)
            msg.setDefaultButton(QtWidgets.QMessageBox.Save)
            x = msg.exec_()
            if x == QtWidgets.QMessageBox.Save: self.saveFile()
            else: self.statusbar.showMessage('Data discarded')
        else:
            self.saveFile()

    def saveFile(self):
        try:
            checkFile = os.path.join(self.saveDirectory, self.fileName + '.mat')
            i = 1
            while os.path.isfile(checkFile):
                newFileName = self.fileName + str(i)
                checkFile = os.path.join(self.saveDirectory, newFileName + '.mat')
                i += 1
            
            dataDictionary = {'emgRaw': self.emgRaw, 'fs_EMG': self.params['fs'], 'params': self.params}
            savemat(checkFile, dataDictionary)
            self.statusbar.showMessage('Saved!')
        except Exception:
            self.statusbar.showMessage('Something went wrong')

    def fcn_stop_stim(self):
        commnadSize = 25
        self.stop = 0
        stimFreqDataSet = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100])
        stimDurDataSet = np.array([100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 750, 800, 850, 900, 950, 1000])

        confString = [commnadSize, 0, 0]
        stimFreqIndex = np.argmin(abs(stimFreqDataSet - self.params['stimFrequency'])) + 1
        confString.append(32 + stimFreqIndex)
        stimDurIndex = np.argmin(abs(stimDurDataSet - self.params['pulseWidth'])) + 1
        confString.append(stimDurIndex)

        for _ in range(18): confString.append(0)

        confString.append(self.CRC8(confString, commnadSize - 1))
        self.serialPort.write(struct.pack("{0}B".format(len(confString)), *confString))
        commandSent = 1
        self.stop = 1
        return commandSent

    def fcn_start_stim(self):
        commnadSize = 25
        self.stop = 0
        stimFreqDataSet = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100])
        stimDurDataSet = np.array([100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 750, 800, 850, 900, 950, 1000])
        stimAmpDataSet = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

        confString = [commnadSize, 0]
        if self.params['fs'] == 16000: confString[-1] += int('00110000', 2)
        elif self.params['fs'] == 2000: confString[-1] += int('00000000', 2)
        elif self.params['fs'] == 4000: confString[-1] += int('00010000', 2)
        elif self.params['fs'] == 8000: confString[-1] += int('00100000', 2)

        if self.params['emgChannels'] == 2: confString[-1] += int('00000000', 2)
        elif self.params['emgChannels'] == 4: confString[-1] += int('00000100', 2)
        elif self.params['emgChannels'] == 8: confString[-1] += int('00001000', 2)
        elif self.params['emgChannels'] == 16: confString[-1] += int('00001100', 2)

        confString[-1] += 1
        confString.append(0)

        for i in range(8):
            if self.params['stimAmplitude'][i] > 0: confString[-1] += (1 << i)

        stimFreqIndex = np.argmin(abs(stimFreqDataSet - self.params['stimFrequency'])) + 1
        stimType = 3 * 32 if self.params['stimPhase'] == 'Biphasic' else (1 * 32 if self.params['stimPhase'] == 'Positive' else 2 * 32)
        confString.append(stimType + stimFreqIndex)

        stimDurIndex = np.argmin(abs(stimDurDataSet - self.params['pulseWidth'])) + 1
        confString.append(stimDurIndex)

        for i in range(8):
            stimAmpIndex = np.argmin(abs(stimAmpDataSet - self.params['stimAmplitude'][i])) + 1
            confString.append(stimAmpIndex)

        confString.append(self.params['timeON'])
        confString.append(self.params['timeOFF'])
        confString.append(self.params['timeRead'])
        confString.append(self.params['timeStim'])
        confString.append(self.params['trialSeq'])

        confString.append(0)
        confString.append(self.params['tremorThreshold'])
        confString.append(self.params['gain1'])
        confString.append(self.params['gain2'])
        confString.append(self.params['timeRMS'])
        confString.append((self.params['ecrChannel'] << 4) | self.params['fcrChannel'])

        confString.append(self.CRC8(confString, commnadSize - 1))
        sendConfstring = [int(x) for x in confString]
        self.serialPort.write(struct.pack("{0}B".format(len(sendConfstring)), *sendConfstring))

        self.readEMG(display_channels=[self.params['ecrChannel'], self.params['fcrChannel']])

        commandSent = 1
        self.stop = 0
        return commandSent

    def CRC8(self, vector, lenVector):
        crc = 0
        j = 0
        while lenVector > 0:
            Extract = vector[j]
            for i in range(8, 0, -1):
                dummy = int(Extract) % 2 if Extract % 1 == 0 else 1
                Sum = operator.xor(crc % 2, dummy)
                crc = crc // 2
                if Sum > 0:
                    st = np.zeros((8))
                    a = bin(crc)[2:].zfill(8)
                    b = bin(140)[2:].zfill(8)
                    for k in range(8): st[k] = not ((a[k] == b[k]))
                    crc = int(np.array2string(st).replace('.', '').replace(' ', '').replace('[', '').replace(']', ''), 2)
                Extract = Extract // 2
            lenVector -= 1
            j += 1
        return crc

    def onAnalyzeButtonClicked(self):
        fileFilter = 'Data File (*.mat)'
        self.dataFilePath = str(QtWidgets.QFileDialog.getOpenFileName(self, 'Select File to Analyze', directory=self.saveDirectory, filter=fileFilter)[0])
        self.statusbar.showMessage(('Loading ' + self.dataFilePath))
        emgData = loadmat(self.dataFilePath)
        self.mainSATS(emgData)

    def mainSATS(self, emgData):
        try: self.plotWindow.close()
        except Exception: pass
        self.getParameters()
        fs = float(emgData['fs_EMG'][0][0])
        order = 200
        lpCutoff = 10
        wn = (lpCutoff) / (fs / 2)
        fir_bLP = signal.firwin(order + 1, wn, window='hamming', pass_zero='lowpass')

        emgRaw = emgData['emgRaw'].T
        ecr = self.params['ecrChannel']
        fcr = self.params['fcrChannel']
        emgRaw = emgRaw[:, [ecr, fcr]]

        trialLength = max(emgRaw.shape)
        windowRecON = np.zeros([trialLength,])
        recWindow = int(round(self.params['timeRead'] * fs))
        stimWindow = int(round(self.params['timeStim'] * fs))
        
        muscle = 0
        tremorThreshold = self.params['tremorThreshold']
        emgRMSThreshold = np.array([0.004, 0.002])
        currentRMS = np.array([0.004, 0.002])
        stimDuration = 0.25
        windowRMS = int(round(self.params['timeRMS'] / 1000 * fs))
        gain1 = self.params['gain1'] / 10
        gain2 = self.params['gain2'] / 10

        state = 0
        stimulationOutput = np.zeros([trialLength, 2])
        burstDetected = []
        burstPredicted = []
        currentPosition = 0
        plotMax = 250
        w = int(1 * fs)
        k = 0

        while w < trialLength:
            if state == 0: state = 1
            elif state == 1:
                windowRecON[w - 1:w + recWindow] = plotMax
                currentEMG = emgRaw[w - 1:w + recWindow, :]
                emgEnvelope = signal.lfilter(fir_bLP, 1, np.abs(currentEMG), axis=0)

                emgRMSThreshold[0] = np.sqrt(np.mean(currentEMG[:, 0]**2)) * gain1
                emgRMSThreshold[1] = np.sqrt(np.mean(currentEMG[:, 1]**2)) * gain2

                tremorBurst = np.zeros([len(emgEnvelope),])
                tremorBurstLoc = []
                for j in range(1, len(emgEnvelope) - 1):
                    if emgEnvelope[j, muscle] > tremorThreshold and emgEnvelope[j, muscle] < 300 and emgEnvelope[j, muscle] > emgEnvelope[j + 1, muscle] and emgEnvelope[j, muscle] > emgEnvelope[j - 1, muscle]:
                        tremorBurstLoc.append(round(j - order / 2))

                tremorBurst[[round(x) for x in tremorBurstLoc]] = 30
                nBursts = len(tremorBurstLoc)

                ibi = []
                if nBursts > 4:
                    state = 2
                    for i in range(nBursts - 1): ibi.append(abs(tremorBurstLoc[i] - tremorBurstLoc[i + 1]))
                    ibiMean = mean(ibi) / fs
                    nextBurst = []
                    j = 1
                    while round(tremorBurstLoc[-1] + ibiMean * j * fs) < stimWindow:
                        nextBurst.append(round(tremorBurstLoc[-1] + ibiMean * j * fs))
                        j += 1
                    burstDetected.extend([x + w for x in tremorBurstLoc])
                    burstPredicted.extend([x + w for x in nextBurst])
                else:
                    w = w + recWindow 
                    state = 1

            elif state == 2:
                if k < trialLength:
                    w = w + recWindow
                    k = w
                    while k < w + stimWindow - round(windowRMS):
                        currentEMG = emgRaw[k - 1:round(k + windowRMS), :]
                        currentRMS[0] = np.sqrt(np.mean(currentEMG[:, 0]**2))
                        currentRMS[1] = np.sqrt(np.mean(currentEMG[:, 1]**2))
                        currentPosition = round(k + windowRMS)

                        if currentRMS[1] > emgRMSThreshold[1]:
                            stimulationOutput[currentPosition - 1:round(currentPosition + fs * ibiMean * stimDuration), 0] = plotMax
                        if currentRMS[0] > emgRMSThreshold[0]:
                            stimulationOutput[currentPosition - 1:round(currentPosition + fs * ibiMean * stimDuration), 1] = -plotMax

                        if currentRMS[0] > emgRMSThreshold[0] or currentRMS[1] > emgRMSThreshold[1]:
                            k = round(currentPosition + fs * ibiMean * stimDuration)
                        else:
                            k = currentPosition + 1
                    w = w + stimWindow
                    state = 1
                else:
                    w = trialLength + 1
                    state = 0

        emgFilEnvelope = signal.lfilter(fir_bLP, 1, np.abs(emgRaw), axis=0)
        t = np.arange(0, (len(emgFilEnvelope)) / fs, 1 / fs)

        burstDetectedSignal = np.zeros([len(emgRaw),])
        burstPredictedSignal = np.zeros([len(emgRaw),])
        burstDetectedSignal[burstDetected] = plotMax
        burstPredictedSignal[[x for x in burstPredicted if x < len(emgRaw)]] = plotMax

        self.plotWindow = MatplotlibWindow()
        self.plotWindow.show()
        self.plotWindow.plot(t, emgRaw, emgFilEnvelope, burstDetectedSignal, burstPredictedSignal, windowRecON, stimulationOutput, order)

    def closeEvent(self, event):
        try: self.extraWindow.close()
        except Exception: pass
        try: self.plotWindow.close()
        except Exception: pass
        if self.calib_plot_window is not None:
            try: self.calib_plot_window.close()
            except Exception: pass
        event.accept()

class MatplotlibWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setStyleSheet('background-color: white')
        self.setWindowIcon(QtGui.QIcon(':/icons/Icons/lightning-bolt.png'))
        self.setWindowTitle('Plot')
        self.setLayout(layout)

    def plot(self, t, emgRaw, emgFilEnvelope, burstDetectedSignal, burstPredictedSignal, windowRecON, stimulationOutput, order):  
        self.figure.clear()
        plt.plot(t, emgRaw[:, 0], label='ECR')
        plt.plot(t, emgRaw[:, 1], label='FCR')
        plt.plot(t[0:round(len(t) - order / 2 + 1)], emgFilEnvelope[round(order / 2) - 1:, 0], label='ECR envelope')
        plt.plot(t[0:round(len(t) - order / 2 + 1)], emgFilEnvelope[round(order / 2) - 1:, 1], label='FCR envelope')
        plt.plot(t, burstDetectedSignal, label='Tremor detected')
        plt.plot(t, burstPredictedSignal, label='Tremor predicted')
        plt.plot(t, stimulationOutput[:, 0], label='Stimulation ECR', linestyle='-.')
        plt.plot(t, stimulationOutput[:, 1], label='Stimulation FCR', linestyle='-.')
        plt.plot(t, windowRecON, label='Recording window', linestyle='--')

        plt.legend()
        plt.xlabel('s')
        plt.ylabel('mV')
        plt.subplots_adjust(top=0.981, bottom=0.075, left=0.057, right=0.99, hspace=0.2, wspace=0.2)
        self.canvas.draw()

class AnotherWindow(QtWidgets.QWidget):
    def __init__(self, num_ch=2):
        super().__init__()
        self.num_ch = num_ch

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)

        container = QtWidgets.QWidget()
        cols = PLOT_COLS
        rows = (num_ch + cols - 1) // cols
        grid = QtWidgets.QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(2)

        pen_emg = pg.mkPen(color=(0, 0, 255), width=1.5)
        pen_env = pg.mkPen(color=(242, 90, 44), width=2)

        self.plots = []
        self.data_lines = []
        self.data_adds = []

        ch_labels = ['ECR', 'FCR'] + [f'Ch {i+1}' for i in range(2, num_ch)]

        for i in range(num_ch):
            pw = pg.PlotWidget()
            pw.setBackground('w')
            pw.setLabel('left', 'mV', size='8pt')
            pw.setLabel('bottom', 's', size='8pt')
            pw.setTitle(ch_labels[i], size='8pt')
            pw.setYRange(-2500, 2500, padding=0)

            row_i, col_i = divmod(i, cols)
            grid.addWidget(pw, row_i, col_i)

            line = pw.plot([], [], pen=pen_emg)
            line_add = pw.plot([], [], pen=pen_env)

            self.plots.append(pw)
            self.data_lines.append(line)
            self.data_adds.append(line_add)

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

        self.setLayout(main_layout)
        self.setStyleSheet('background-color: white')
        self.setWindowIcon(QtGui.QIcon(':/icons/Icons/lightning-bolt.png'))
        self.setWindowTitle('Plot')

        w = 1000
        h = min(200 * rows, 900)
        self.setGeometry(100, 100, w, h)

        self.x = list(range(100))
        self.x_add = np.zeros((100,))
        self.y_data = [np.zeros((100,)) for _ in range(num_ch)]
        self.y_add_data = [np.zeros((100,)) for _ in range(num_ch)]
        self.n_plot = 0

    def updateSeparatePlot(self, x_to_plot, y_to_plot, additional_x, additional_y):
        if self.n_plot == 0:
            for i in range(self.num_ch): self.y_data[i] = y_to_plot[i]
            self.x = x_to_plot
            self.x_add = additional_x
            for i in range(len(additional_y)): self.y_add_data[i] = additional_y[i]
        else:
            for i in range(self.num_ch): self.y_data[i] = np.append(self.y_data[i], y_to_plot[i])
            for i in range(len(additional_y)): self.y_add_data[i] = additional_y[i]

        self.n_plot = len(self.y_data[0])
        plot_len = len(self.x)

        for i in range(self.num_ch):
            if self.n_plot < plot_len:
                self.data_lines[i].setData(self.x[0:self.n_plot], self.y_data[i][0:self.n_plot])
            else:
                self.data_lines[i].setData(self.x, self.y_data[i][self.n_plot - plot_len:])

            if i < len(additional_y):
                env = self.y_add_data[i]
                self.data_adds[i].setData(self.x_add[0:len(env)], env)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    sys.exit(app.exec_())