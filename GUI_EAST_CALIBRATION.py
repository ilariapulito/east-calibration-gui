from PyQt5 import QtWidgets, uic, QtGui, QtCore
from PyQt5.QtWidgets import QMessageBox
import PyQt5
import sys

import os
import serial
import numpy as np
import operator
import re
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
    run_stim_calibration,
    evaluate_stim_round,
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


class EmgRecordingWorker(QtCore.QThread):
    signal_chunk = QtCore.pyqtSignal(object)
    signal_complete = QtCore.pyqtSignal(object, object, str)
    signal_error = QtCore.pyqtSignal(str)

    def __init__(self, serial_port, params, file_name, stop_callback, parent=None):
        super().__init__(parent)
        self.serial_port = serial_port
        self.params = dict(params)
        self.file_name = file_name
        self.stop_callback = stop_callback
        self.is_running = True
        self.emg_chunks = []

    def stop(self):
        self.is_running = False

    def run(self):
        num_channels_total = self.params["emgChannels"] + 1
        samples_per_chunk = 125
        bytes_per_chunk = 2 * num_channels_total * samples_per_chunk
        conv_factor = 0.000286 * 1000
        offsets = None
        num_cycles = int(np.ceil(
            self.params["recordingTime"] * self.params["fs"] / samples_per_chunk
        ))

        try:
            self.serial_port.reset_input_buffer()
            while self.is_running and (
                num_cycles == 0 or len(self.emg_chunks) < num_cycles
            ):
                if self.serial_port.in_waiting < bytes_per_chunk:
                    self.msleep(5)
                    continue

                chunk_bytes = self.serial_port.read(bytes_per_chunk)
                if not chunk_bytes:
                    raise RuntimeError("Serial connection closed during EMG acquisition.")

                current_data = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float64)
                current_data = current_data.reshape(
                    (num_channels_total, samples_per_chunk), order="F"
                )
                spurious_index = int(np.argmin(np.var(current_data, axis=1)))
                shift_back = (num_channels_total - 1 - spurious_index) % num_channels_total
                if shift_back:
                    current_data = np.roll(current_data, shift_back, axis=0)

                if offsets is None:
                    spurious_index = int(np.argmin(np.var(current_data, axis=1)))
                    offsets = np.zeros(num_channels_total)
                    emg_positions = [
                        index for index in range(num_channels_total)
                        if index != spurious_index
                    ]
                    for position, offset in zip(emg_positions, OFFSET_BRACELET_CH):
                        offsets[position] = offset

                self.emg_chunks.append(current_data)
                plot_data = (
                    current_data[:self.params["emgChannels"]] * conv_factor
                    - offsets[:self.params["emgChannels"], np.newaxis]
                )
                self.signal_chunk.emit(plot_data)
        except Exception as error:
            self.signal_error.emit(str(error))
        finally:
            try:
                self.stop_callback(self.params)
            except Exception as error:
                self.signal_error.emit(f"Could not stop EMG acquisition cleanly: {error}")

            if self.emg_chunks:
                emg_raw = np.concatenate(self.emg_chunks, axis=1)
                emg_raw = emg_raw * conv_factor - offsets[:, np.newaxis]
            else:
                emg_raw = np.empty((num_channels_total, 0))
            self.signal_complete.emit(emg_raw, self.params, self.file_name)


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
        self.stim_calib_plot_window = None
        self.calib_worker = None
        self.calib_emg_chunks = []
        self.calib_mode_active = None  # 'read' or 'stim'
        self.emg_recording_worker = None

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
            self.ch16radioButton.setChecked(True)
            self.stackedWidget.setCurrentWidget(self.calibration_page)
        elif hasattr(self, 'stimulationCalibrationButton') and self.stimulationCalibrationButton.isChecked():
            self.ch16radioButton.setChecked(True)
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
            QWidget#stimulationCalibrationPage { background-color: #f3f6fb; }
            QWidget#stimulationCalibrationPage QGroupBox {
                background-color: white; border: 1px solid #d5deeb;
                border-radius: 8px; margin-top: 10px; padding: 8px;
                font-weight: bold; color: #23395d;
            }
            QWidget#stimulationCalibrationPage QSpinBox,
            QWidget#stimulationCalibrationPage QDoubleSpinBox {
                min-height: 26px; border: 1px solid #c8d3e2;
                border-radius: 4px; padding: 2px 6px; background: white;
            }
            QWidget#stimulationCalibrationPage QPushButton {
                min-height: 34px; border: none; border-radius: 5px;
                padding: 4px 12px; background: #e4ebf5;
                color: #23395d; font-weight: bold;
            }
            QPushButton#startStimulationCalibrationButton {
                background: #198754; color: white;
            }
            QPushButton#stopStimulationCalibrationButton {
                background: #dc3545; color: white;
                min-height: 50px; font-size: 15px;
            }
            QLabel#stimGuidanceBanner {
                background: #e8f1ff; border: 1px solid #9bbcf0;
                border-radius: 8px; padding: 14px; color: #183b68;
                font-size: 15px; font-weight: bold;
            }
        """)

        root = QtWidgets.QVBoxLayout(self.stimulation_calibration_page)
        root.setContentsMargins(18, 14, 18, 18)
        root.setSpacing(10)

        header = QtWidgets.QHBoxLayout()
        back_button = QtWidgets.QPushButton("← Return to Main GUI")
        back_button.clicked.connect(self.exit_calibration_mode)
        title = QtWidgets.QLabel("STIMULATION CALIBRATION")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #23395d;")
        header.addWidget(back_button)
        header.addWidget(title)
        header.addStretch()
        root.addLayout(header)

        columns = QtWidgets.QHBoxLayout()
        columns.setSpacing(14)
        root.addLayout(columns, stretch=1)
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(8)
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(8)
        columns.addLayout(left, stretch=35)
        columns.addLayout(right, stretch=65)

        channels_box = QtWidgets.QGroupBox("1. Calibrated Read Channels")
        channels_form = QtWidgets.QFormLayout(channels_box)
        self.stimulationCalibrationEcrChannel = self.copy_spin_box(
            self.ecrChannel, channels_box, "stimulationCalibrationEcrChannel"
        )
        self.stimulationCalibrationFcrChannel = self.copy_spin_box(
            self.fcrChannel, channels_box, "stimulationCalibrationFcrChannel"
        )
        for control in (self.stimulationCalibrationEcrChannel,
                        self.stimulationCalibrationFcrChannel):
            control.setRange(1, 16)
        self.stimulationCalibrationEcrChannel.valueChanged.connect(self.ecrChannel.setValue)
        self.ecrChannel.valueChanged.connect(self.stimulationCalibrationEcrChannel.setValue)
        self.stimulationCalibrationFcrChannel.valueChanged.connect(self.fcrChannel.setValue)
        self.fcrChannel.valueChanged.connect(self.stimulationCalibrationFcrChannel.setValue)
        channels_form.addRow("ECR Channel (1-16)", self.stimulationCalibrationEcrChannel)
        channels_form.addRow("FCR Channel (1-16)", self.stimulationCalibrationFcrChannel)
        left.addWidget(channels_box)

        params_box = QtWidgets.QGroupBox("2. Electrode & Current Round Parameters")
        params_form = QtWidgets.QFormLayout(params_box)
        self.stimTotalElectrodesSpin = QtWidgets.QSpinBox(params_box)
        self.stimTotalElectrodesSpin.setRange(2, 8)
        self.stimTotalElectrodesSpin.setValue(6)
        self.stimGiriSpin = QtWidgets.QSpinBox(params_box)
        self.stimGiriSpin.setRange(1, 10)
        self.stimGiriSpin.setValue(2)
        self.stimStartCurrentSpin = QtWidgets.QDoubleSpinBox(params_box)
        self.stimStartCurrentSpin.setRange(0.5, 20.0)
        self.stimStartCurrentSpin.setSingleStep(0.5)
        self.stimStartCurrentSpin.setDecimals(1)
        self.stimStartCurrentSpin.setValue(2.0)
        self.stimCurrentStepSpin = QtWidgets.QDoubleSpinBox(params_box)
        self.stimCurrentStepSpin.setRange(0.5, 10.0)
        self.stimCurrentStepSpin.setSingleStep(0.5)
        self.stimCurrentStepSpin.setDecimals(1)
        self.stimCurrentStepSpin.setValue(1.0)
        self.stimMaxRoundsSpin = QtWidgets.QSpinBox(params_box)
        self.stimMaxRoundsSpin.setRange(1, 15)
        self.stimMaxRoundsSpin.setValue(5)
        self.stimulationCalibrationPulseWidth = self.copy_spin_box(
            self.pulseWidthBox, params_box, "stimulationCalibrationPulseWidth"
        )
        self.stimulationCalibrationIsi = QtWidgets.QDoubleSpinBox(params_box)
        self.stimulationCalibrationIsi.setRange(0.5, 10.0)
        self.stimulationCalibrationIsi.setSingleStep(0.5)
        self.stimulationCalibrationIsi.setDecimals(1)
        self.stimulationCalibrationIsi.setValue(1.0)
        self.stimulationCalibrationIsi.setSuffix(" s")
        for widget, suffix in (
            (self.stimStartCurrentSpin, " mA"),
            (self.stimCurrentStepSpin, " mA"),
        ):
            widget.setSuffix(suffix)
        params_form.addRow("Total Electrodes", self.stimTotalElectrodesSpin)
        params_form.addRow("Turns per Current (Giri)", self.stimGiriSpin)
        params_form.addRow("Start Current (mA)", self.stimStartCurrentSpin)
        params_form.addRow("Current Step (mA)", self.stimCurrentStepSpin)
        params_form.addRow("Max Current Steps", self.stimMaxRoundsSpin)
        params_form.addRow("Pulse Width (µs)", self.stimulationCalibrationPulseWidth)
        params_form.addRow("ISI (s)", self.stimulationCalibrationIsi)

        self.stimulationCalibrationPhaseButtons = {}
        self.stimulationCalibrationPhaseGroup = QtWidgets.QButtonGroup(self)
        phase_row = QtWidgets.QWidget(params_box)
        phase_layout = QtWidgets.QHBoxLayout(phase_row)
        phase_layout.setContentsMargins(0, 0, 0, 0)
        for key, label in (
            ("biphasicButton", "Biphasic"),
            ("positiveButton", "Positive"),
            ("negativeButton", "Negative"),
        ):
            button = QtWidgets.QRadioButton(label, phase_row)
            button.setObjectName(f"stimulationCalibration{key}")
            button.setChecked(getattr(self, key).isChecked())
            self.stimulationCalibrationPhaseGroup.addButton(button)
            self.stimulationCalibrationPhaseButtons[key] = button
            phase_layout.addWidget(button)
        params_form.addRow("Stimulation Phase", phase_row)
        left.addWidget(params_box)

        self.startStimCalibBtn = QtWidgets.QPushButton("Start Stimulation Calibration")
        self.startStimCalibBtn.setObjectName("startStimulationCalibrationButton")
        self.startStimCalibBtn.setEnabled(False)
        self.startStimCalibBtn.clicked.connect(self.start_stim_calibration)
        self.stopStimCalibBtn = QtWidgets.QPushButton("STOP / Uncomfortable")
        self.stopStimCalibBtn.setObjectName("stopStimulationCalibrationButton")
        self.stopStimCalibBtn.setEnabled(False)
        self.stopStimCalibBtn.clicked.connect(self.stop_stim_calibration)
        left.addWidget(self.startStimCalibBtn)
        left.addWidget(self.stopStimCalibBtn)
        left.addStretch(1)

        self.stimGuidanceBanner = QtWidgets.QLabel(
            "Ready. Arm pronated, flat & relaxed."
        )
        self.stimGuidanceBanner.setObjectName("stimGuidanceBanner")
        self.stimGuidanceBanner.setWordWrap(True)
        self.stimStatusLabel = self.stimGuidanceBanner
        right.addWidget(self.stimGuidanceBanner)

        cards = QtWidgets.QHBoxLayout()
        self.lbl_rad_status = QtWidgets.QGroupBox("Radial Nerve (ECR — 1st Half)")
        radial_layout = QtWidgets.QVBoxLayout(self.lbl_rad_status)
        self.lbl_rad_res = QtWidgets.QLabel("Waiting...")
        self.lbl_rad_res.setWordWrap(True)
        radial_layout.addWidget(self.lbl_rad_res)
        self.lbl_med_status = QtWidgets.QGroupBox("Median Nerve (FCR — 2nd Half)")
        median_layout = QtWidgets.QVBoxLayout(self.lbl_med_status)
        self.lbl_med_res = QtWidgets.QLabel("Waiting...")
        self.lbl_med_res.setWordWrap(True)
        median_layout.addWidget(self.lbl_med_res)
        cards.addWidget(self.lbl_rad_status)
        cards.addWidget(self.lbl_med_status)
        right.addLayout(cards)

        plots_toolbar = QtWidgets.QHBoxLayout()
        plots_toolbar.addStretch()
        self.toggleStimPlotsBtn = QtWidgets.QPushButton("Open Assistive Plots")
        self.toggleStimPlotsBtn.setCheckable(True)
        self.toggleStimPlotsBtn.setChecked(False)
        self.toggleStimPlotsBtn.setVisible(False)
        self.toggleStimPlotsBtn.toggled.connect(self.toggle_stim_assistive_plots)
        plots_toolbar.addWidget(self.toggleStimPlotsBtn)
        right.addLayout(plots_toolbar)

        self.stim_aux_panel = QtWidgets.QGroupBox("Assistive Plots")
        stim_panel_layout = QtWidgets.QVBoxLayout(self.stim_aux_panel)
        scroll = QtWidgets.QScrollArea(self.stim_aux_panel)
        scroll.setWidgetResizable(True)
        plot_content = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_content)
        self.stim_radial_grid_widget = pg.GraphicsLayoutWidget()
        self.stim_radial_grid_widget.setMinimumHeight(300)
        self.stim_median_grid_widget = pg.GraphicsLayoutWidget()
        self.stim_median_grid_widget.setMinimumHeight(300)
        self.pw_stim_raw_stacked = pg.PlotWidget()
        self.pw_stim_raw_stacked.setMinimumHeight(360)
        self.pw_stim_raw_stacked.setBackground("w")
        for caption, widget in (
            ("Epoch grid 5–30 ms — Radial / Extensor (ECR)", self.stim_radial_grid_widget),
            ("Epoch grid 5–30 ms — Median / Flexor (FCR)", self.stim_median_grid_widget),
            ("16 Raw Channels (Stacked)", self.pw_stim_raw_stacked),
        ):
            plot_layout.addWidget(QtWidgets.QLabel(caption))
            plot_layout.addWidget(widget)
        scroll.setWidget(plot_content)
        stim_panel_layout.addWidget(scroll)
        self.stim_aux_panel.setVisible(False)
        right.addWidget(self.stim_aux_panel, stretch=1)

        self.startStimulationCalibrationButton = self.startStimCalibBtn
        self.stopStimulationCalibrationButton = self.stopStimCalibBtn
        self.stim_round_timer = QtCore.QTimer(self)
        self.stim_round_timer.setSingleShot(True)
        self.stim_round_timer.timeout.connect(self.finish_stimulation_round)
        self.stim_round_grids = {"radial": None, "median": None}
        self.stim_calibration_state = None

    def setup_legacy_stimulation_calibration_page_ui(self):
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

        sweep_box = QtWidgets.QGroupBox("Stimulation Calibration Sweep")
        sweep_layout = QtWidgets.QVBoxLayout(sweep_box)
        sweep_buttons = QtWidgets.QHBoxLayout()
        self.startStimCalibBtn = QtWidgets.QPushButton("Start Stimulation Calibration")
        self.startStimCalibBtn.setEnabled(False)
        self.startStimCalibBtn.clicked.connect(self.start_stim_calibration)
        self.stopStimCalibBtn = QtWidgets.QPushButton("Stop Stimulation Calibration")
        self.stopStimCalibBtn.setEnabled(False)
        self.stopStimCalibBtn.setObjectName("stopStimCalibBtn")
        self.stopStimCalibBtn.setStyleSheet(
            "background-color: #dc3545; color: white; font-weight: bold;"
        )
        self.stopStimCalibBtn.clicked.connect(self.stop_stim_calibration)
        sweep_buttons.addWidget(self.startStimCalibBtn)
        sweep_buttons.addWidget(self.stopStimCalibBtn)
        sweep_layout.addLayout(sweep_buttons)

        self.stimStatusLabel = QtWidgets.QLabel(
            "Stimulation Status: Complete Lecture Calibration to enable."
        )
        sweep_layout.addWidget(self.stimStatusLabel)
        self.aux_tab_widget = QtWidgets.QTabWidget()
        self.tab_stim_plot = QtWidgets.QWidget()
        stim_plot_layout = QtWidgets.QVBoxLayout(self.tab_stim_plot)
        self.pw_stim_response = pg.PlotWidget()
        self.pw_stim_response.setBackground("w")
        self.pw_stim_response.setTitle(
            "Evoked Responses per Electrode (ECR vs FCR)"
        )
        self.pw_stim_response.setLabel("bottom", "Stimulation Electrode")
        self.pw_stim_response.setLabel("left", "Response RMS")
        self.pw_stim_response.addLegend()
        stim_plot_layout.addWidget(self.pw_stim_response)
        self.aux_tab_widget.addTab(
            self.tab_stim_plot, "Stimulation Calibration Analysis"
        )
        sweep_layout.addWidget(self.aux_tab_widget)
        layout.addWidget(sweep_box)

        self.lbl_rad_res = QtWidgets.QLabel("Radial Electrode: Unassigned")
        self.lbl_med_res = QtWidgets.QLabel("Median Electrode: Unassigned")
        sweep_layout.addWidget(self.lbl_rad_res)
        sweep_layout.addWidget(self.lbl_med_res)

    def setup_calibration_page_ui(self):
        """Builds the lecture-calibration controls and assistive plots."""
        self.calibration_page.setObjectName("lectureCalibrationPage")
        self.calibration_page.setStyleSheet("""
            QWidget#lectureCalibrationPage {
                background-color: #f3f6fb;
            }
            QWidget#lectureCalibrationPage QGroupBox {
                background-color: #ffffff;
                border: 1px solid #d5deeb;
                border-radius: 8px;
                margin-top: 10px;
                padding: 10px;
                font-weight: bold;
                color: #23395d;
            }
            QWidget#lectureCalibrationPage QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QWidget#lectureCalibrationPage QPushButton {
                min-height: 34px;
                border: none;
                border-radius: 5px;
                padding: 4px 12px;
                background-color: #e4ebf5;
                color: #23395d;
                font-weight: bold;
            }
            QWidget#lectureCalibrationPage QPushButton:hover {
                background-color: #d3deed;
            }
            QWidget#lectureCalibrationPage QPushButton#startReadCalibBtn {
                background-color: #198754;
                color: #ffffff;
            }
            QWidget#lectureCalibrationPage QPushButton#stopReadCalibBtn {
                background-color: #dc3545;
                color: #ffffff;
            }
        """)

        main_layout = QtWidgets.QVBoxLayout(self.calibration_page)
        main_layout.setContentsMargins(18, 14, 18, 18)
        main_layout.setSpacing(12)

        columns = QtWidgets.QHBoxLayout()
        columns.setSpacing(14)
        main_layout.addLayout(columns)

        left_column = QtWidgets.QVBoxLayout()
        left_column.setSpacing(12)
        columns.addLayout(left_column, stretch=1)

        self.btn_back_to_standard = QtWidgets.QPushButton("← Return to Main GUI")
        self.btn_back_to_standard.clicked.connect(self.exit_calibration_mode)
        title_label = QtWidgets.QLabel("LECTURE CALIBRATION")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #23395d;")
        left_column.addWidget(self.btn_back_to_standard)
        left_column.addWidget(title_label)

        proto_box = QtWidgets.QGroupBox("Protocol Configuration")
        proto_layout = QtWidgets.QVBoxLayout(proto_box)
        self.protocolInputLabel = QtWidgets.QLabel(
            "Protocol sequence (0 = FCR, 1 = ECR):"
        )
        self.protocolLineEdit = QtWidgets.QLineEdit("1, 1, 0, 0, 1, 1, 0, 0")
        proto_layout.addWidget(self.protocolInputLabel)
        proto_layout.addWidget(self.protocolLineEdit)
        left_column.addWidget(proto_box)

        btn_read_layout = QtWidgets.QHBoxLayout()
        self.startReadCalibBtn = QtWidgets.QPushButton("Start EMG Sampling")
        self.startReadCalibBtn.setObjectName("startReadCalibBtn")
        self.startReadCalibBtn.setStyleSheet("background-color: #0d6efd; color: white; font-weight: bold; padding: 6px;")
        self.startReadCalibBtn.clicked.connect(self.start_read_calibration)
        self.stopReadCalibBtn = QtWidgets.QPushButton("Stop EMG Sampling")
        self.stopReadCalibBtn.setObjectName("stopReadCalibBtn")
        self.stopReadCalibBtn.setEnabled(False)
        self.stopReadCalibBtn.clicked.connect(self.stop_read_calibration)
        btn_read_layout.addWidget(self.startReadCalibBtn)
        btn_read_layout.addWidget(self.stopReadCalibBtn)
        left_column.addLayout(btn_read_layout)

        feedback_box = QtWidgets.QGroupBox("Calibration Feedback")
        feedback_layout = QtWidgets.QVBoxLayout(feedback_box)
        self.readFeedbackLabel = QtWidgets.QLabel(
            "Ready. Enter protocol and press Start EMG Sampling."
        )
        self.readFeedbackLabel.setWordWrap(True)
        self.set_read_feedback("ready")
        feedback_layout.addWidget(self.readFeedbackLabel)
        left_column.addWidget(feedback_box)
        left_column.addStretch(1)

        right_column = QtWidgets.QVBoxLayout()
        right_column.setSpacing(10)
        columns.addLayout(right_column, stretch=2)

        self.assistive_plot_data = None
        self.assistive_plot_windows = {}
        self.assistivePlotsPanel = QtWidgets.QGroupBox("Assistive Plots")
        plots_layout = QtWidgets.QVBoxLayout(self.assistivePlotsPanel)
        plots_layout.addWidget(
            QtWidgets.QLabel("Select a plot to open it in an interactive window:")
        )
        self.assistivePlotButtons = {}
        assistive_plot_options = (
            ("raw", "16 Raw Channels (Stacked)"),
            ("envelope", "Mean RMS Envelope & Activation Windows"),
            ("filtered", "16 Filtered Channels — Top 3 ECR / FCR"),
        )
        for plot_key, plot_title in assistive_plot_options:
            button = QtWidgets.QPushButton(plot_title)
            button.setEnabled(False)
            button.clicked.connect(
                lambda checked=False, key=plot_key: self.open_assistive_plot(key)
            )
            plots_layout.addWidget(button)
            self.assistivePlotButtons[plot_key] = button
        self.assistivePlotsPanel.setVisible(True)
        right_column.addWidget(self.assistivePlotsPanel)

        selected_box = QtWidgets.QGroupBox("Selected Best Channels")
        selected_layout = QtWidgets.QVBoxLayout(selected_box)
        self.lbl_ecr_res = QtWidgets.QLabel("ECR Channel: Unassigned")
        self.lbl_fcr_res = QtWidgets.QLabel("FCR Channel: Unassigned")
        for label in (self.lbl_ecr_res, self.lbl_fcr_res):
            label.setStyleSheet("font-size: 15px; font-weight: bold; color: #23395d;")
            selected_layout.addWidget(label)
        right_column.addWidget(selected_box)

    def set_read_feedback(self, state, message=None):
        messages = {
            "ready": "Ready. Enter protocol and press Start EMG Sampling.",
            "success": "Calibration successful! Movement count matches protocol.",
        }
        colors = {
            "ready": ("#f1f5f9", "#64748b", "#cbd5e1"),
            "success": ("#e8f5ee", "#198754", "#a3d9b8"),
            "error": ("#fff1f0", "#b42318", "#f0a39b"),
        }
        background, foreground, border = colors[state]
        text = message if message is not None else messages[state]
        self.readFeedbackLabel.setText(text)
        self.readFeedbackLabel.setStyleSheet(
            f"padding: 10px; border: 1px solid {border}; border-radius: 6px; "
            f"background-color: {background}; color: {foreground};"
        )

    def open_assistive_plot(self, plot_key):
        if self.assistive_plot_data is None:
            return

        existing_window = self.assistive_plot_windows.get(plot_key)
        if existing_window is not None:
            existing_window.show()
            existing_window.raise_()
            existing_window.activateWindow()
            return

        plot_titles = {
            "raw": "16 Raw Channels (Stacked)",
            "envelope": "Mean RMS Envelope & Activation Windows",
            "filtered": "16 Filtered Channels — Top 3 ECR (Green) & Top 3 FCR (Red)",
        }
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(plot_titles[plot_key])
        dialog.resize(1100, 700)
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        dialog_layout = QtWidgets.QVBoxLayout(dialog)
        toolbar = QtWidgets.QHBoxLayout()
        reset_button = QtWidgets.QPushButton("Reset Zoom")
        toolbar.addWidget(reset_button)
        toolbar.addStretch()
        dialog_layout.addLayout(toolbar)

        plot_widget = pg.PlotWidget()
        plot_widget.setBackground("w")
        plot_widget.setMinimumHeight(500)
        dialog_layout.addWidget(plot_widget)
        reset_button.clicked.connect(plot_widget.autoRange)

        self.render_assistive_plot(plot_widget, plot_key, self.assistive_plot_data)
        self.assistive_plot_windows[plot_key] = dialog
        dialog.finished.connect(
            lambda _result, key=plot_key: self.assistive_plot_windows.pop(key, None)
        )
        dialog.show()

    def exit_calibration_mode(self):
        if self.stim_calibration_state is not None:
            self.stop_stim_calibration()
        if hasattr(self, 'manualButton'):
            self.manualButton.setChecked(True)

    def send_start_EMG_command(self):
        """Invia il pacchetto seriale di configurazione a 25 byte al firmware STM32."""
        commnadSize = 25
        stimFreqDataSet = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100])
        stimDurDataSet = np.array([100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 750, 800, 850, 900, 950, 1000])
        stimAmpDataSet = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

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
        active_mask = 0
        if self.params.get("stimulationCalibration"):
            for index, current in enumerate(self.params["stimAmplitude"]):
                if current > 0:
                    active_mask |= 1 << index
        confString.append(active_mask)

        stimFreqIndex = np.argmin(abs(stimFreqDataSet - self.params['stimFrequency'])) + 1
        stimType = 3*32 if self.params['stimPhase'] == 'Biphasic' else (1*32 if self.params['stimPhase'] == 'Positive' else 2*32)
        confString.append(stimType + stimFreqIndex)

        stimDurIndex = np.argmin(abs(stimDurDataSet - self.params['pulseWidth'])) + 1
        confString.append(stimDurIndex)

        for current in self.params.get("stimAmplitude", [0.0] * 8):
            amp_index = int(np.argmin(abs(stimAmpDataSet - current))) + 1
            confString.append(amp_index if current > 0 else 0)

        confString.append(self.params['timeON'])
        confString.append(self.params['timeOFF'])
        confString.append(self.params['timeRead'])
        confString.append(self.params['timeStim'])
        confString.append(self.params['trialSeq'])
        confString.append(3 if self.params.get("stimulationCalibration") else 0)
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
        self.set_read_feedback("ready", "EMG sampling in progress. Press Stop EMG Sampling when finished.")

        if self.calib_plot_window is not None:
            self.calib_plot_window.close()
        self.calib_plot_window = AnotherWindow(num_ch=16)
        self.calib_plot_window.setWindowTitle("Lecture Calibration - 16 EMG Channels")
        self.calib_plot_window.show()

        self.calib_worker = CalibrationWorker(ui=self)
        self.calib_worker.signal_chunk.connect(self.on_calibration_chunk_received)
        self.calib_worker.signal_error.connect(self.on_calibration_error)
        self.calib_worker.start()

    def on_calibration_chunk_received(self, chunk_16ch: np.ndarray):
        """Store each acquisition chunk and draw it in the standard 16-channel grid."""
        if (
            self.calib_mode_active == "stim_round"
            and self.sender() is not self.calib_worker
        ):
            return
        self.calib_emg_chunks.append(chunk_16ch)

        if self.calib_mode_active == 'read' and self.calib_plot_window is not None:
            fs = self.params.get('fs', 2000)
            time_vec = np.linspace(0, 1, fs)
            self.calib_plot_window.updateSeparatePlot(
                x_to_plot=time_vec,
                y_to_plot=chunk_16ch,
                additional_x=time_vec,
                additional_y=[],
            )
        elif (
            self.calib_mode_active == "stim_round"
            and self.stim_calib_plot_window is not None
            and self.stim_calib_plot_window.isVisible()
        ):
            fs = self.params.get("fs", 2000)
            ecr_channel = self.params["ecrChannel"]
            fcr_channel = self.params["fcrChannel"]
            time_vec = np.linspace(0, 1, fs)
            self.stim_calib_plot_window.updateSeparatePlot(
                x_to_plot=time_vec,
                y_to_plot=[
                    chunk_16ch[ecr_channel, :],
                    chunk_16ch[fcr_channel, :],
                ],
                additional_x=time_vec,
                additional_y=[],
            )

    def stop_read_calibration(self):
        """Stops sampling, saves the recording, and runs lecture calibration."""
        if self.calib_worker is not None:
            worker = self.calib_worker
            worker.stop()
            QtWidgets.QApplication.processEvents()
            if self.calib_worker is worker:
                self.calib_worker = None

        self.stopReadCalibBtn.setEnabled(False)
        self.startReadCalibBtn.setEnabled(True)

        if len(self.calib_emg_chunks) == 0:
            self.set_read_feedback("error", "No EMG data was recorded. Please repeat the sampling.")
            return

        emg_raw_16ch = np.hstack(self.calib_emg_chunks)
        fs = self.params.get('fs', 2000)
        protocol_str = self.protocolLineEdit.text()
        try:
            protocol = parse_protocol_string(protocol_str)
        except ValueError as e:
            self.save_lecture_calibration_recording(
                emg_raw_16ch, fs, protocol_str
            )
            self.set_read_feedback("error", str(e))
            return

        self.save_lecture_calibration_recording(emg_raw_16ch, fs, protocol)

        try:
            ecr_ch, fcr_ch, aux_plot_data = run_read_calibration(
                emg_raw_16ch=emg_raw_16ch,
                protocol=protocol,
                fs=fs
            )
            self.on_read_calibration_success(ecr_ch, fcr_ch, aux_plot_data)
        except MovementMismatchError as e:
            self.update_assistive_plots(e.aux_plot_data)
            self.on_read_calibration_mismatch(e)
        except Exception as e:
            self.on_calibration_error(str(e))

    def save_lecture_calibration_recording(self, emg_raw_16ch, fs, protocol):
        run_number = 1
        date_prefix = date.today().strftime("%Y%m%d")
        while True:
            file_name = f"{date_prefix}_lecture_calib_run_{run_number:02d}.mat"
            file_path = os.path.join(self.saveDirectory, file_name)
            if not os.path.exists(file_path):
                break
            run_number += 1

        try:
            savemat(
                file_path,
                {
                    "emgRaw": emg_raw_16ch,
                    "fs_EMG": fs,
                    "protocol": protocol,
                    "params": self.params,
                },
            )
            self.statusbar.showMessage(f"Lecture calibration recording saved: {file_path}")
        except (OSError, ValueError, TypeError) as e:
            QMessageBox.critical(
                self,
                "Recording Save Error",
                f"Could not save the lecture calibration recording:\n{e}",
            )

    def on_read_calibration_success(self, ecr_ch, fcr_ch, aux_plot_data):
        self.calibrated_ecr = ecr_ch
        self.calibrated_fcr = fcr_ch
        self.update_selected_channel_controls(ecr_ch, fcr_ch)
        self.lbl_ecr_res.setText(f"ECR Channel: Ch {ecr_ch + 1}")
        self.lbl_fcr_res.setText(f"FCR Channel: Ch {fcr_ch + 1}")
        self.update_assistive_plots(aux_plot_data)
        self.set_read_feedback("success")
        self.startStimCalibBtn.setEnabled(True)
        self.stimStatusLabel.setText("Stimulation Status: Ready for execution")

    def update_selected_channel_controls(self, ecr_ch, fcr_ch):
        channel_values = (
            ("ecrChannel", ecr_ch),
            ("fcrChannel", fcr_ch),
            ("stimulationCalibrationEcrChannel", ecr_ch),
            ("stimulationCalibrationFcrChannel", fcr_ch),
        )
        for attribute, channel in channel_values:
            control = getattr(self, attribute, None)
            if control is not None:
                control.setValue(channel + 1)

    def reset_selected_channel_controls(self):
        self.calibrated_ecr = None
        self.calibrated_fcr = None
        for attribute in (
            "ecrChannel",
            "fcrChannel",
            "stimulationCalibrationEcrChannel",
            "stimulationCalibrationFcrChannel",
        ):
            control = getattr(self, attribute, None)
            if control is not None:
                control.setValue(1)
        self.lbl_ecr_res.setText("ECR Channel: Unassigned")
        self.lbl_fcr_res.setText("FCR Channel: Unassigned")

    def update_assistive_plots(self, aux_plot_data):
        self.assistive_plot_data = aux_plot_data
        for button in self.assistivePlotButtons.values():
            button.setEnabled(True)

        for plot_key, dialog in list(self.assistive_plot_windows.items()):
            plot_widget = dialog.findChild(pg.PlotWidget)
            if plot_widget is not None:
                self.render_assistive_plot(plot_widget, plot_key, aux_plot_data)

    def render_assistive_plot(self, plot_widget, plot_key, aux_plot_data):
        time_vec = np.asarray(aux_plot_data.get("time", []))
        raw = np.asarray(aux_plot_data.get("emg_raw_16ch", np.empty((0, 0))))
        filtered = np.asarray(aux_plot_data.get("emg_filt_16ch", np.empty((0, 0))))
        if raw.ndim != 2 or raw.shape[0] != 16:
            raw = np.empty((16, 0))
        if filtered.ndim != 2 or filtered.shape[0] != 16:
            filtered = np.empty((16, 0))

        if time_vec.size != raw.shape[1] and raw.shape[1]:
            fs = self.params.get("fs", 2000)
            time_vec = np.arange(raw.shape[1]) / fs
        if time_vec.size != filtered.shape[1] and filtered.shape[1]:
            fs = self.params.get("fs", 2000)
            filtered_time = np.arange(filtered.shape[1]) / fs
        else:
            filtered_time = time_vec

        plot_widget.clear()
        plot_widget.setLabel("bottom", "Time", units="s")

        if plot_key in ("raw", "filtered"):
            is_raw = plot_key == "raw"
            samples = raw if is_raw else filtered
            plot_widget.setTitle(
                "16 Raw Channels (Stacked)"
                if is_raw
                else "16 Filtered Channels — Top 3 ECR (Green) & Top 3 FCR (Red)"
            )
            plot_widget.setLabel("left", "Amplitude (mV)")
            scale = max(
                5 * float(np.std(samples)) if samples.size else 0.0,
                float(np.max(np.ptp(samples, axis=1))) if samples.size else 0.0,
                0.5,
            )
            ticks = [
                ((15 - channel) * scale, f"Ch {channel + 1}")
                for channel in range(16)
            ]
            if not is_raw:
                top3_ecr = set(aux_plot_data.get("top3_ecr", []))
                top3_fcr = set(aux_plot_data.get("top3_fcr", []))
                ticks = []
                for channel in range(16):
                    labels = []
                    if channel in top3_ecr:
                        labels.append("ECR")
                    if channel in top3_fcr:
                        labels.append("FCR")
                    suffix = f" ({'/'.join(labels)})" if labels else ""
                    ticks.append(
                        ((15 - channel) * scale, f"Ch {channel + 1}{suffix}")
                    )

            plot_widget.getAxis("left").setTicks([ticks])
            if samples.shape[1]:
                sample_time = time_vec if is_raw else filtered_time
                top3_ecr = set(aux_plot_data.get("top3_ecr", []))
                top3_fcr = set(aux_plot_data.get("top3_fcr", []))
                for channel in range(16):
                    if is_raw:
                        color, width = "#4b5563", 1.0
                    elif channel in top3_ecr:
                        color, width = "#198754", 2.0
                    elif channel in top3_fcr:
                        color, width = "#dc3545", 2.0
                    else:
                        color, width = "#c4cbd4", 1.0
                    plot_widget.plot(
                        sample_time,
                        samples[channel] + (15 - channel) * scale,
                        pen=pg.mkPen(color, width=width),
                    )
            return

        plot_widget.setTitle("Mean RMS Envelope & Activation Windows")
        plot_widget.setLabel("left", "Amplitude (mV)")
        env_mean = np.asarray(aux_plot_data.get("env_mean", []))
        if not env_mean.size:
            return

        if time_vec.size != env_mean.size:
            fs = self.params.get("fs", 2000)
            env_time = np.arange(env_mean.size) / fs
        else:
            env_time = time_vec
        plot_widget.plot(
            env_time,
            env_mean,
            pen=pg.mkPen("#fd7e14", width=2),
        )
        threshold = float(aux_plot_data.get("threshold", 0))
        plot_widget.plot(
            env_time,
            np.full(env_mean.shape, threshold),
            pen=pg.mkPen("#dc3545", width=1.5, style=QtCore.Qt.DashLine),
        )
        for onset, offset in zip(
            aux_plot_data.get("onsets", []),
            aux_plot_data.get("offsets", []),
        ):
            region = pg.LinearRegionItem(
                values=[onset, offset],
                brush=(25, 135, 84, 45),
                movable=False,
            )
            region.setZValue(-10)
            plot_widget.addItem(region)

    def on_read_calibration_mismatch(self, error):
        self.reset_selected_channel_controls()
        self.calibrated_ecr = None
        self.calibrated_fcr = None
        self.startReadCalibBtn.setEnabled(True)
        self.stopReadCalibBtn.setEnabled(False)
        self.startStimCalibBtn.setEnabled(False)
        self.stimStatusLabel.setText(
            "Stimulation Status: Disabled until lecture calibration succeeds."
        )
        self.set_read_feedback("error", str(error))

    # ── LOGICA CALIBRAZIONE STIMOLAZIONE ────────────────────────────────────
    def start_stim_calibration(self):
        """Starts the sequential radial and median nerve calibration."""
        if not hasattr(self, 'serialPort') or self.serialPort is None or not self.serialPort.is_open:
            QMessageBox.critical(self, "Error", "Hardware EAST non connesso!")
            return

        if self.calibrated_ecr is None or self.calibrated_fcr is None:
            QMessageBox.critical(self, "Error", "Completare prima la Calibrazione di Lettura.")
            return

        self.getParameters()
        isi_s = float(self.stimulationCalibrationIsi.value())
        if not isi_s.is_integer():
            QMessageBox.warning(
                self, "Unsupported ISI",
                "The connected firmware calibration mode supports whole-second ISI values only. "
                "Choose 1, 2, ... 10 seconds.",
            )
            return

        total_electrodes = self.stimTotalElectrodesSpin.value()
        half = total_electrodes // 2
        start_current = self.stimStartCurrentSpin.value()
        current_step = self.stimCurrentStepSpin.value()
        max_rounds = self.stimMaxRoundsSpin.value()
        scheduled_currents = [
            start_current + index * current_step
            for index in range(max_rounds)
        ]
        if scheduled_currents[-1] > 20.0:
            QMessageBox.warning(
                self, "Current range",
                "The selected current steps exceed the device maximum of 20 mA.",
            )
            return
        supported_currents = np.array([
            0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5,
            7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20,
        ])
        if any(
            not np.any(np.isclose(current, supported_currents))
            for current in scheduled_currents
        ):
            QMessageBox.warning(
                self, "Unsupported current",
                "The device does not support every 0.5 mA value above 11 mA. "
                "Adjust the start current or step so each round uses a supported current.",
            )
            return
        if self.params["ecrChannel"] == self.params["fcrChannel"]:
            QMessageBox.warning(
                self, "Invalid channels",
                "ECR and FCR must use different EMG channels.",
            )
            return

        self.stim_round_grids = {"radial": None, "median": None}
        self.stim_calibration_state = {
            "radial_els": list(range(1, half + 1)),
            "median_els": list(range(half + 1, total_electrodes + 1)),
            "total_electrodes": total_electrodes,
            "n_giri": self.stimGiriSpin.value(),
            "start_current": start_current,
            "current_step": current_step,
            "max_rounds": max_rounds,
            "isi_s": isi_s,
            "pulse_width": self.stimulationCalibrationPulseWidth.value(),
            "phase": self.params["stimPhase"],
            "ecr_channel": self.params["ecrChannel"],
            "fcr_channel": self.params["fcrChannel"],
            "nerve": "radial",
            "round_idx": 0,
            "current_ma": start_current,
        }
        self.calibrated_radial_el = None
        self.calibrated_median_el = None
        self.lbl_rad_res.setText("Searching...")
        self.lbl_med_res.setText("Waiting...")
        self.stim_radial_grid_widget.clear()
        self.stim_radial_grid_widget.addLabel("No evaluated round yet", row=0, col=0)
        self.stim_median_grid_widget.clear()
        self.stim_median_grid_widget.addLabel("No evaluated round yet", row=0, col=0)
        self.pw_stim_raw_stacked.clear()
        self.pw_stim_raw_stacked.setTitle("16 Raw Channels (Stacked)")
        self.toggleStimPlotsBtn.setChecked(False)
        self.startStimCalibBtn.setEnabled(False)
        self.stopStimCalibBtn.setEnabled(True)
        self.toggleStimPlotsBtn.setVisible(True)
        if self.stim_calib_plot_window is not None:
            self.stim_calib_plot_window.close()
        self.stim_calib_plot_window = AnotherWindow(num_ch=2)
        self.stim_calib_plot_window.setWindowTitle(
            "Stimulation Calibration - ECR/FCR EMG"
        )
        self.stim_calib_plot_window.show()
        self._start_stimulation_round()

    def _start_stimulation_round(self):
        state = self.stim_calibration_state
        if state is None:
            return
        nerve = state["nerve"]
        active_els = state[f"{nerve}_els"]
        current_ma = state["current_ma"]
        round_number = state["round_idx"] + 1
        role = "Radial Nerve (ECR)" if nerve == "radial" else "Median Nerve (FCR)"
        channel_range = f"El. {active_els[0]}–{active_els[-1]}"
        self.params.update({
            "stimulationCalibration": True,
            "mode": "CON",
            "emgChannels": 16,
            "stimAmplitude": [
                current_ma if electrode in active_els else 0.0
                for electrode in range(1, 9)
            ],
            "pulseWidth": state["pulse_width"],
            "stimFrequency": state["isi_s"],
            "stimPhase": state["phase"],
            "ecrChannel": state["ecr_channel"],
            "fcrChannel": state["fcr_channel"],
        })
        self.calib_emg_chunks = []
        self.calib_mode_active = "stim_round"
        if nerve == "radial":
            self.lbl_rad_res.setText("Searching...")
        else:
            self.lbl_med_res.setText("Searching...")
        self.stimGuidanceBanner.setText(
            f"Testing {role} ({channel_range}) | Round {round_number}/"
            f"{state['max_rounds']} at {current_ma:.1f} mA — "
            "Stay still while pulsing..."
        )
        self.calib_worker = CalibrationWorker(ui=self)
        self.calib_worker.signal_chunk.connect(self.on_calibration_chunk_received)
        self.calib_worker.signal_error.connect(self.on_stimulation_round_error)
        self.calib_worker.start()
        duration_ms = int(
            (len(active_els) * state["n_giri"] + 1) * state["isi_s"] * 1000
        )
        self.stim_round_timer.start(duration_ms)

    def stop_stim_calibration(self):
        """Immediately aborts the current round and sends the firmware stop command."""
        if self.stim_calibration_state is None:
            return
        state = self.stim_calibration_state
        nerve = state["nerve"]
        active_els = state[f"{nerve}_els"]
        self._stop_stimulation_worker()
        if self.calib_emg_chunks:
            self.save_stimulation_round(
                np.hstack(self.calib_emg_chunks),
                self.params.get("fs", 2000),
                state,
                nerve,
                active_els,
                {"found": False, "message": "Stopped by user before round completed."},
            )
        self.stim_calibration_state = None
        self.calib_mode_active = None
        self.stopStimCalibBtn.setEnabled(False)
        self.startStimCalibBtn.setEnabled(
            self.calibrated_ecr is not None and self.calibrated_fcr is not None
        )
        self.stimGuidanceBanner.setText(
            "STOPPED by user (Uncomfortable / Paused)."
        )

    def _stop_stimulation_worker(self):
        if self.stim_round_timer.isActive():
            self.stim_round_timer.stop()
        if self.calib_worker is not None:
            worker = self.calib_worker
            if getattr(self, "serialPort", None) is not None and self.serialPort.is_open:
                worker.stop_command_sent = True
                self.fcn_stop_stim()
            worker.stop()
            self.calib_worker = None

    def on_stimulation_round_error(self, err_msg):
        if self.stim_calibration_state is not None:
            self._stop_stimulation_worker()
            self.stim_calibration_state = None
            self.calib_mode_active = None
            self.stopStimCalibBtn.setEnabled(False)
            self.startStimCalibBtn.setEnabled(
                self.calibrated_ecr is not None and self.calibrated_fcr is not None
            )
            self.stimGuidanceBanner.setText(f"Calibration error: {err_msg}")
        QMessageBox.critical(self, "Stimulation Calibration Error", err_msg)

    def finish_stimulation_round(self):
        state = self.stim_calibration_state
        if state is None:
            return
        self._stop_stimulation_worker()
        emg_raw_16ch = (
            np.hstack(self.calib_emg_chunks)
            if self.calib_emg_chunks else np.empty((16, 0), dtype=float)
        )
        fs = self.params.get("fs", 2000)
        nerve = state["nerve"]
        active_els = state[f"{nerve}_els"]
        target_ch = state["ecr_channel"] if nerve == "radial" else state["fcr_channel"]
        other_ch = state["fcr_channel"] if nerve == "radial" else state["ecr_channel"]
        role_name = "Extensor" if nerve == "radial" else "Flexor"
        try:
            if emg_raw_16ch.shape[1] == 0:
                result = {
                    "found": False, "chosen_el": None, "chosen_ch": None,
                    "current_ma": state["current_ma"],
                    "message": "No EMG samples were recorded in this round.",
                    "grid_data": None,
                }
            else:
                result = evaluate_stim_round(
                    emg_raw_16ch=emg_raw_16ch,
                    target_ch=target_ch,
                    other_main_ch=other_ch,
                    active_electrodes_1based=active_els,
                    current_ma=state["current_ma"],
                    n_giri=state["n_giri"],
                    fs=fs,
                    isi_s=state["isi_s"],
                    role_name=role_name,
                )
        except Exception as e:
            self.save_stimulation_round(
                emg_raw_16ch, fs, state, nerve, active_els,
                {"found": False, "evaluationError": str(e)},
            )
            self.on_stimulation_round_error(str(e))
            return

        self.save_stimulation_round(
            emg_raw_16ch, fs, state, nerve, active_els, result
        )
        grid_data = result.get("grid_data")
        self.stim_round_grids[nerve] = grid_data
        self.update_stimulation_assistive_plots(grid_data, emg_raw_16ch, fs, state)

        if result["found"]:
            chosen_el = result["chosen_el"]
            chosen_ch = result["chosen_ch"]
            if nerve == "radial":
                self.calibrated_radial_el = chosen_el
                self.lbl_rad_res.setText(
                    f"LOCKED: El. {chosen_el} at {state['current_ma']:.1f} mA "
                    f"(Ch {chosen_ch + 1})"
                )
                state.update({
                    "nerve": "median",
                    "round_idx": 0,
                    "current_ma": state["start_current"],
                })
                self.lbl_med_res.setText("Searching...")
                self._start_stimulation_round()
                return
            self.calibrated_median_el = chosen_el
            self.lbl_med_res.setText(
                f"LOCKED: El. {chosen_el} at {state['current_ma']:.1f} mA "
                f"(Ch {chosen_ch + 1})"
            )
            self._finish_stimulation_calibration(
                "Calibration complete. Radial and median responses locked."
            )
            return

        next_round_idx = state["round_idx"] + 1
        next_current = state["start_current"] + next_round_idx * state["current_step"]
        if next_round_idx < state["max_rounds"] and next_current <= 20.0:
            state["round_idx"] = next_round_idx
            state["current_ma"] = next_current
            self._start_stimulation_round()
        elif nerve == "radial":
            self.lbl_rad_res.setText("Not found (Rotate cuff)")
            state.update({
                "nerve": "median",
                "round_idx": 0,
                "current_ma": state["start_current"],
            })
            self.lbl_med_res.setText("Searching...")
            self._start_stimulation_round()
        else:
            self.lbl_med_res.setText("Not found (Rotate cuff)")
            self._finish_stimulation_calibration(
                "Maximum current reached without selective response. "
                "Rotate cuff if needed."
            )

    def _finish_stimulation_calibration(self, message):
        self._stop_stimulation_worker()
        self.stim_calibration_state = None
        self.calib_mode_active = None
        self.stopStimCalibBtn.setEnabled(False)
        self.startStimCalibBtn.setEnabled(
            self.calibrated_ecr is not None and self.calibrated_fcr is not None
        )
        self.stimGuidanceBanner.setText(message)

    def save_stimulation_round(self, emg_raw_16ch, fs, state, nerve, active_els, result):
        date_prefix = date.today().strftime("%Y%m%d")
        current = f"{state['current_ma']:.1f}"
        electrodes = "-".join(str(el) for el in active_els)
        base_name = (
            f"{date_prefix}_stim_calib_{nerve}_r{state['round_idx'] + 1}_"
            f"{current}mA_els{electrodes}_pw{state['pulse_width']:g}us_"
            f"isi{state['isi_s']:g}s"
        )
        run_number = 1
        while True:
            suffix = "" if run_number == 1 else f"_run{run_number:02d}"
            file_path = os.path.join(
                self.saveDirectory, base_name + suffix + ".mat"
            )
            if not os.path.exists(file_path):
                break
            run_number += 1
        try:
            data = {
                "emgRaw": emg_raw_16ch,
                "fs_EMG": fs,
                "params": self.params,
                "nerve": nerve,
                "activeElectrodes": active_els,
                "round": state["round_idx"] + 1,
                "current_mA": state["current_ma"],
                "isi_s": state["isi_s"],
                "n_giri": state["n_giri"],
                "found": bool(result.get("found", False)),
                "chosenElectrode": result.get("chosen_el") or 0,
                "chosenChannel": (
                    result["chosen_ch"] + 1
                    if result.get("chosen_ch") is not None else 0
                ),
                "message": result.get("message", result.get("evaluationError", "")),
            }
            grid_data = result.get("grid_data")
            if grid_data is not None:
                data["stimIdx"] = grid_data["stim_idx"]
                data["pred"] = grid_data["pred"]
                data["scoreRF"] = grid_data["score_rf"]
                data["scoreSV"] = grid_data["score_sv"]
                data["peakToPeak"] = grid_data["pp"]
            savemat(file_path, data)
            self.statusbar.showMessage(f"Stimulation round saved: {file_path}")
        except (OSError, ValueError, TypeError, KeyError) as e:
            QMessageBox.critical(
                self, "Round Save Error",
                f"Could not save stimulation round:\n{e}",
            )

    def toggle_stim_assistive_plots(self, is_open):
        self.stim_aux_panel.setVisible(is_open)
        self.toggleStimPlotsBtn.setText(
            "Hide Assistive Plots" if is_open else "Open Assistive Plots"
        )

    def update_stimulation_assistive_plots(self, grid_data, raw, fs, state):
        nerve = state["nerve"]
        widget = (
            self.stim_radial_grid_widget
            if nerve == "radial" else self.stim_median_grid_widget
        )
        self.render_stimulation_epoch_grid(
            widget,
            grid_data,
            "radial" if nerve == "radial" else "median",
        )
        self.render_stimulation_raw_plot(grid_data, raw, fs, state)

    def render_stimulation_epoch_grid(self, widget, grid_data, nerve):
        widget.clear()
        if not grid_data:
            widget.addLabel("No evaluated round yet", row=0, col=0)
            return

        active_els = grid_data["active_electrodes_1based"]
        n_giri = int(grid_data["n_giri"])
        times_ms = np.asarray(grid_data["t_win_ms"])
        epochs = grid_data["epochs"]
        stim_electrodes = np.asarray(grid_data["el_1based_of"])
        turns = np.asarray(grid_data["giro_of"])
        valid_mask = np.asarray(grid_data["valid_mask"], dtype=bool)
        predictions = np.asarray(grid_data["pred"], dtype=bool)
        score_rf = np.asarray(grid_data["score_rf"])
        score_sv = np.asarray(grid_data["score_sv"])
        peak_to_peak = np.asarray(grid_data["pp"])
        chosen_el = grid_data.get("chosen_el")
        candidate_electrodes = {
            candidate["el_1based"]
            for candidate in grid_data.get("candidates", [])
        }
        selected_channel_index = grid_data.get("chosen_k")
        if selected_channel_index is None:
            selected_channel_index = 0
        selected_channel_index = min(selected_channel_index, epochs.shape[0] - 1)

        selected_color = "#198754" if nerve == "radial" else "#dc3545"
        for row in range(n_giri):
            for col, electrode in enumerate(active_els):
                plot = widget.addPlot(row=row, col=col)
                sample_indices = np.where(
                    (stim_electrodes == electrode) & (turns == row + 1)
                )[0]
                chosen_sample = (
                    int(sample_indices[0]) if sample_indices.size else None
                )
                is_valid = (
                    electrode in candidate_electrodes
                    and
                    chosen_sample is not None
                    and valid_mask[chosen_sample]
                    and predictions[selected_channel_index, chosen_sample]
                )
                plot.getViewBox().setBackgroundColor(
                    "#d9f7be" if is_valid else "#eeeeee"
                )
                plot.addItem(pg.InfiniteLine(
                    pos=10,
                    angle=90,
                    pen=pg.mkPen("#596579", style=QtCore.Qt.DashLine, width=1),
                ))
                if chosen_sample is None or not valid_mask[chosen_sample]:
                    plot.setTitle(f"El.{electrode} r{row + 1} | no epoch")
                else:
                    rf_value = score_rf[selected_channel_index, chosen_sample]
                    sv_value = score_sv[selected_channel_index, chosen_sample]
                    pp_value = peak_to_peak[selected_channel_index, chosen_sample]
                    plot.setTitle(
                        f"El.{electrode} r{row + 1} | "
                        f"RF={rf_value:.2f} SV={sv_value:.2f} PP={pp_value:.2f}",
                        size="8pt",
                    )
                    if chosen_el == electrode:
                        color = selected_color
                        width = 2.5
                    elif is_valid:
                        color = "#1677ff"
                        width = 1.8
                    else:
                        color = "#596579"
                        width = 1.0
                    plot.plot(
                        times_ms,
                        epochs[selected_channel_index, chosen_sample, :],
                        pen=pg.mkPen(color, width=width),
                    )
                plot.setXRange(5, 30, padding=0)
                plot.setLabel("bottom", "ms")
                if col == 0:
                    plot.setLabel("left", f"Turn {row + 1}")
                else:
                    plot.hideAxis("left")

    def render_stimulation_raw_plot(self, grid_data, raw, fs, state):
        plot = self.pw_stim_raw_stacked
        plot.clear()
        plot.setBackground("w")
        plot.setLabel("bottom", "Time", units="s")
        plot.setLabel("left", "Amplitude + channel offset", units="mV")
        plot.setTitle("16 Raw Channels (Stacked)")
        raw = np.asarray(raw)
        if raw.ndim != 2 or raw.shape[0] != 16 or raw.shape[1] == 0:
            plot.setTitle("16 Raw Channels (Stacked) — No samples")
            return

        step = max(
            5 * float(np.std(raw)),
            float(np.max(np.ptp(raw, axis=1))),
            0.5,
        )
        time_s = np.arange(raw.shape[1]) / fs
        ecr_ch = state["ecr_channel"]
        fcr_ch = state["fcr_channel"]
        radial_neighbors = set()
        median_neighbors = set()
        for nerve, main_channel in (("radial", ecr_ch), ("median", fcr_ch)):
            nerve_grid = self.stim_round_grids.get(nerve)
            if nerve_grid:
                analyzed = nerve_grid.get("analysed_channels", [])
                neighbors = set(analyzed) - {main_channel}
                if nerve == "radial":
                    radial_neighbors = neighbors
                else:
                    median_neighbors = neighbors
        for ch in range(16):
            if ch == ecr_ch or ch in radial_neighbors:
                color = "#198754" if ch == ecr_ch else "#71b88f"
            elif ch == fcr_ch or ch in median_neighbors:
                color = "#dc3545" if ch == fcr_ch else "#e58a91"
            else:
                color = "#aeb7c2"
            offset = (15 - ch) * step
            plot.plot(
                time_s,
                raw[ch, :] + offset,
                pen=pg.mkPen(color, width=1.5 if ch in (ecr_ch, fcr_ch) else 0.9),
            )
        ticks = [
            ((15 - ch) * step, f"Ch {ch + 1}")
            for ch in range(16)
        ]
        plot.getAxis("left").setTicks([ticks])
        if grid_data:
            for stim_sample in grid_data.get("stim_idx", []):
                plot.addItem(pg.InfiniteLine(
                    pos=float(stim_sample) / fs,
                    angle=90,
                    pen=pg.mkPen("#303846", style=QtCore.Qt.DashLine, width=1),
                ))

    def confirm_stimulation_data_save(self):
        if self.stimulationCalibrationSaveCheckBox.isChecked():
            return True

        message = QtWidgets.QMessageBox(self)
        message.setIcon(QtWidgets.QMessageBox.Warning)
        message.setWindowTitle("Stimulation Calibration")
        message.setText("Do you want to save the stimulation calibration data?")
        message.setInformativeText("It will be saved in " + self.saveDirectory)
        message.setStandardButtons(
            QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard
        )
        message.setDefaultButton(QtWidgets.QMessageBox.Save)
        return message.exec_() == QtWidgets.QMessageBox.Save

    def build_stimulation_file_path(self, params, recording_kind):
        date_prefix = date.today().strftime("%Y%m%d")
        label = os.path.splitext(os.path.basename(self.fileName.strip()))[0]
        if label.startswith(date_prefix + "_"):
            label = label[len(date_prefix) + 1:]
        label = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_")[:48]

        active_currents = [
            f"ch{channel}-{self.format_filename_value(current)}mA"
            for channel, current in enumerate(params.get("stimAmplitude", []), start=1)
            if current > 0
        ]
        current_tag = "I" + ("_".join(active_currents) if active_currents else "0mA")
        pulse_width_tag = f"PW{self.format_filename_value(params['pulseWidth'])}us"
        if params.get("stimulationCalibration"):
            interval_tag = f"ISI{self.format_filename_value(params['stimFrequency'])}s"
            mode_tag = "stim_calib"
        else:
            interval_tag = f"Freq{self.format_filename_value(params['stimFrequency'])}Hz"
            mode_tag = "stim"
        phase_tag = re.sub(
            r"[^A-Za-z0-9_-]+", "", params.get("stimPhase", "unknown")
        ).lower()
        ecr_tag = f"ECRch{params['ecrChannel'] + 1}"
        fcr_tag = f"FCRch{params['fcrChannel'] + 1}"
        name_parts = [
            date_prefix,
            label,
            recording_kind,
            mode_tag,
            current_tag,
            pulse_width_tag,
            interval_tag,
            phase_tag,
            ecr_tag,
            fcr_tag,
            f"seq{params.get('trialSeq', 0):02X}",
        ]
        if recording_kind == "recording":
            name_parts.append(
                f"rec{self.format_filename_value(params.get('recordingTime', 0))}s"
            )
        name = "_".join(part for part in name_parts if part)

        run_number = 1
        while True:
            file_name = f"{name}_run_{run_number:02d}.mat"
            file_path = os.path.join(self.saveDirectory, file_name)
            if not os.path.exists(file_path):
                return file_path
            run_number += 1

    @staticmethod
    def format_filename_value(value):
        return f"{float(value):g}".replace(".", "p")

    def save_stimulation_calibration_recording(
        self, emg_raw_16ch, fs, params, calibration_results
    ):
        try:
            file_path = self.build_stimulation_file_path(params, "sweep")
            data = {
                "emgRaw": emg_raw_16ch,
                "fs_EMG": fs,
                "params": params,
            }
            data.update(calibration_results)
            savemat(file_path, data)
            self.statusbar.showMessage(
                f"Stimulation calibration recording saved: {file_path}"
            )
        except (OSError, ValueError, TypeError) as e:
            QMessageBox.critical(
                self,
                "Recording Save Error",
                f"Could not save the stimulation calibration recording:\n{e}",
            )

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

        self.aux_tab_widget.setCurrentIndex(0)

    def on_calibration_error(self, err_msg):
        if self.calib_worker is not None:
            self.calib_worker.stop()
            self.calib_worker = None

        if self.calib_mode_active == "read":
            self.startReadCalibBtn.setEnabled(True)
            self.stopReadCalibBtn.setEnabled(False)
            self.set_read_feedback("error", err_msg)
        elif self.calib_mode_active == "stim":
            self.startStimCalibBtn.setEnabled(self.calibrated_ecr is not None)
            self.stopStimCalibBtn.setEnabled(False)
            self.stimStatusLabel.setText(f"Stimulation Status: Error - {err_msg}")
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
        if self.emg_recording_worker is not None and self.emg_recording_worker.isRunning():
            return
        self.getParameters()
        self.extraWindow = AnotherWindow(num_ch=self.params['emgChannels'])
        self.extraWindow.show()
        self.fcn_start_EMG()

    def onStopEmgButtonClicked(self):
        worker = self.emg_recording_worker
        if worker is not None and worker.isRunning():
            worker.stop()
            self.statusbar.showMessage("Stopping EMG acquisition...")
        else:
            self.fcn_stop_stim()

    def on_emg_recording_chunk(self, chunk):
        plot_window = getattr(self, "extraWindow", None)
        if plot_window is None or plot_window.num_ch != len(chunk):
            return
        fs = self.params["fs"]
        time_vec = np.linspace(0, 1, fs)
        plot_window.updateSeparatePlot(
            x_to_plot=time_vec,
            y_to_plot=chunk,
            additional_x=time_vec,
            additional_y=[],
        )

    def on_emg_recording_error(self, message):
        QMessageBox.critical(self, "EMG Acquisition Error", message)

    def on_emg_recording_complete(self, emg_raw, params, file_name):
        self.params = params
        self.fileName = file_name
        self.stop = 1
        self.emgRaw = emg_raw

        if emg_raw.shape[1] == 0:
            self.statusbar.showMessage("No EMG data was recorded.")
            QMessageBox.warning(self, "EMG Acquisition", "No EMG data was recorded.")
            return

        if not self.params.get("saveData", self.saveCheckBox.isChecked()):
            message = QtWidgets.QMessageBox(self)
            message.setIcon(QtWidgets.QMessageBox.Warning)
            message.setWindowTitle("EAST GUI")
            message.setText("Do you want to save the recorded data?")
            message.setInformativeText("It will be saved in " + self.saveDirectory)
            message.setStandardButtons(
                QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard
            )
            message.setDefaultButton(QtWidgets.QMessageBox.Save)
            if message.exec_() == QtWidgets.QMessageBox.Save:
                self.saveFile()
            else:
                self.statusbar.showMessage("Data discarded")
        else:
            self.saveFile()

    def on_emg_recording_worker_finished(self, worker):
        if self.emg_recording_worker is worker:
            self.emg_recording_worker = None

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
            amplitude_boxes = getattr(self, "stimulationCalibrationChannels", [])
            sequence_boxes = getattr(self, "stimulationTrialSequenceBoxes", [])
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

        self.params['stimAmplitude'] = (
            [box.value() for box in amplitude_boxes]
            if amplitude_boxes else [0.0] * 8
        )

        if sequence_boxes:
            trialSeq = [
                box.isChecked() << index
                for index, box in enumerate(sequence_boxes)
            ]
            self.params['trialSeq'] = sum(trialSeq)
        else:
            self.params['trialSeq'] = 0

        if stimulation_calibration:
            time_on_box = getattr(self, "stimulationCalibrationTimeOn", self.timeON)
            time_off_box = getattr(self, "stimulationCalibrationTimeOff", self.timeOFF)
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

        if self.params.get('mode') == 'Calibration' or stimulation_calibration:
            self.params['emgChannels'] = 16
        else:
            if self.ch2radioButton.isChecked(): self.params['emgChannels'] = 2
            elif self.ch4radioButton.isChecked(): self.params['emgChannels'] = 4
            elif self.ch8radioButton.isChecked(): self.params['emgChannels'] = 8
            elif self.ch16radioButton.isChecked(): self.params['emgChannels'] = 16

        if stimulation_calibration:
            recording_time_box = getattr(self, "stimulationCalibrationRecordingTime", self.recordingTimeBox)
            file_name_edit = getattr(self, "stimulationCalibrationFileName", self.nameEdit)
            save_data_checkbox = getattr(self, "stimulationCalibrationSaveCheckBox", self.saveCheckBox)
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
        if self.emg_recording_worker is not None and self.emg_recording_worker.isRunning():
            raise RuntimeError("EMG acquisition is already running.")

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

        self.stop = 0
        worker = EmgRecordingWorker(
            serial_port=self.serialPort,
            params=self.params,
            file_name=self.fileName,
            stop_callback=self.fcn_stop_stim,
            parent=self,
        )
        worker.signal_chunk.connect(self.on_emg_recording_chunk)
        worker.signal_error.connect(self.on_emg_recording_error)
        worker.signal_complete.connect(self.on_emg_recording_complete)
        worker.finished.connect(
            lambda active_worker=worker: self.on_emg_recording_worker_finished(
                active_worker
            )
        )
        self.emg_recording_worker = worker
        worker.start()
        return 1

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
            if self.params.get("mode") == "CON":
                checkFile = self.build_stimulation_file_path(
                    self.params, "recording"
                )
                self.fileName = os.path.splitext(os.path.basename(checkFile))[0]
            else:
                checkFile = os.path.join(self.saveDirectory, self.fileName + '.mat')
                i = 1
                while os.path.isfile(checkFile):
                    newFileName = self.fileName + str(i)
                    checkFile = os.path.join(self.saveDirectory, newFileName + '.mat')
                    i += 1
            
            dataDictionary = {'emgRaw': self.emgRaw, 'fs_EMG': self.params['fs'], 'params': self.params}
            savemat(checkFile, dataDictionary)
            self.statusbar.showMessage(f"Saved: {checkFile}")
        except (OSError, ValueError, TypeError, KeyError) as e:
            QMessageBox.critical(self, "Recording Save Error", f"Could not save the recording:\n{e}")

    def fcn_stop_stim(self, params=None):
        commnadSize = 25
        self.stop = 0
        params = self.params if params is None else params
        stimFreqDataSet = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100])
        stimDurDataSet = np.array([100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 750, 800, 850, 900, 950, 1000])

        confString = [commnadSize, 0, 0]
        stimFreqIndex = np.argmin(abs(stimFreqDataSet - params['stimFrequency'])) + 1
        confString.append(32 + stimFreqIndex)
        stimDurIndex = np.argmin(abs(stimDurDataSet - params['pulseWidth'])) + 1
        confString.append(stimDurIndex)

        for _ in range(commnadSize - 6): confString.append(0)

        if params.get('stimulationCalibration'):
            confString[18] = 4

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

        # Firmware mode 3 interprets StimFreq as the interval in seconds.
        confString.append(3 if self.params.get('stimulationCalibration') else 0)
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
        if self.stim_calibration_state is not None:
            self.stop_stim_calibration()
        elif self.calib_worker is not None and self.calib_worker.isRunning():
            self.calib_worker.stop()
            self.calib_worker = None
        if self.emg_recording_worker is not None and self.emg_recording_worker.isRunning():
            self.emg_recording_worker.stop()
            self.emg_recording_worker.wait()
        try: self.extraWindow.close()
        except Exception: pass
        if self.stim_calib_plot_window is not None:
            self.stim_calib_plot_window.close()
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
            pw.setClipToView(True)

            row_i, col_i = divmod(i, cols)
            grid.addWidget(pw, row_i, col_i)

            line = pw.plot([], [], pen=pen_emg)
            line.setDownsampling(auto=True, method='peak')
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
        if len(y_to_plot) != self.num_ch:
            raise ValueError(
                f"Expected {self.num_ch} channel traces, received {len(y_to_plot)}."
            )

        if self.n_plot == 0:
            self.x = np.asarray(x_to_plot)
            self.x_add = additional_x
            plot_len = len(self.x)
            self.y_data = [
                np.asarray(y_to_plot[i])[-plot_len:].copy()
                for i in range(self.num_ch)
            ]
            self.y_add_data = [
                np.asarray(additional_y[i]).copy()
                for i in range(len(additional_y))
            ]
            self.n_plot = len(y_to_plot[0]) if self.num_ch else 0
        else:
            plot_len = len(self.x)
            for i in range(self.num_ch):
                combined = np.concatenate((self.y_data[i], np.asarray(y_to_plot[i])))
                self.y_data[i] = combined[-plot_len:]
            for i in range(len(additional_y)):
                combined = np.concatenate(
                    (self.y_add_data[i], np.asarray(additional_y[i]))
                )
                self.y_add_data[i] = combined[-plot_len:]
            self.n_plot += len(y_to_plot[0]) if self.num_ch else 0

        plot_len = len(self.x)

        for i in range(self.num_ch):
            if self.n_plot < plot_len:
                self.data_lines[i].setData(
                    self.x[:len(self.y_data[i])], self.y_data[i]
                )
            else:
                self.data_lines[i].setData(self.x, self.y_data[i])

            if i < len(additional_y):
                env = self.y_add_data[i]
                self.data_adds[i].setData(self.x_add[0:len(env)], env)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    sys.exit(app.exec_())