from PyQt5 import QtWidgets, uic
from PyQt5 import QtGui
from PyQt5.QtCore import QPropertyAnimation
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
from matplotlib.figure import Figure
import pyqtgraph as pg

import resources_EAST #need to compile .qrc file and import output .py file -> pyrcc5 resources.qrc -o resources.py


# ── BRACELET: numero di canali EMG dal braccialetto ──────────────────────────
NUM_BRACELET_CH = 16          # canali EMG fisici del bracelet
PLOT_COLS       = 4           # colonne nella griglia di plot real-time
# ─────────────────────────────────────────────────────────────────────────────

# To bundle app with PyInstaller
try:
    os.chdir(sys._MEIPASS)
except:
    pass

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('GUI_EAST_Qt_v01_names.ui', self)
        self.setWindowIcon(QtGui.QIcon(':/icons/Icons/lightning-bolt.png'))
        # SETUP VARIABLES
        self.fileName = ''
        self.saveDirectory = os.getcwd()
        self.stop = 1

        ports = serial.tools.list_ports.comports()
        listPorts = [p.device for p in ports]
        self.comBox.setEditable(True)

        if listPorts:
            self.comBox.addItems(listPorts)
            self.comBox.setCurrentIndex(0)
        else:
            self.statusbar.showMessage('No COM ports found')
        

        # ----------CONNECTIONS----------
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


        self.show()

    def setFileName(self):
        subject = str(self.subjectBox.value())
        if len(subject) < 2: # to get subjects such as 01 from value = 1
            subject = '0' + subject

        conditon = self.conditionBox.currentText()
        side = self.sideBox.currentText()
        trial = str(self.trialBox.value())
        if len(trial) < 2:
            trial = '0' + trial

        currentDate = str(date.today()).replace('-','')

        # 20240312_subject_03_cond_ASS-PRE-EXT-RIGHT_run_01
        self.fileName = currentDate + '_subject_' + subject + '_cond_' + conditon + '-' + side + '_run_' + trial

        self.nameEdit.setText(self.fileName)


    def onStartStimButtonClicked(self):
        self.getParameters()

        # Modifica bracelet 
        self.extraWindow = AnotherWindow(num_ch=2)
        self.extraWindow.show()
        commandSent = self.fcn_start_stim()

    def onStopStimButtonClicked(self):
        commandSent = self.fcn_stop_stim()


    def onStartEmgButtonClicked(self):
        self.getParameters()
        # ── MODIFICA BRACELET: passa il numero di canali alla finestra di plot ──
        self.extraWindow = AnotherWindow(num_ch=self.params['emgChannels'])
        # ────────────────────────────────────────────────────────────────────────
        #self.extraWindow = AnotherWindow()
        self.extraWindow.show()
        commandSent = self.fcn_start_EMG()


    def onStopEmgButtonClicked(self):
        commandSent = self.fcn_stop_stim()


    def getParameters(self):
        self.params = {}
        
        # MODE
        if self.manualButton.isChecked():
            self.params['mode'] = 'Manual'
        elif self.oopButton.isChecked():
            self.params['mode'] = 'Out-of-phase'
        elif self.satsButton.isChecked():
            self.params['mode'] = 'SATS'
        elif self.conButton.isChecked():
            self.params['mode'] = 'CON'

        # STIMULATION PARAMETERS
        self.params['pulseWidth'] = self.pulseWidthBox.value()
        self.params['stimFrequency'] = self.frequencyBox.value()

        # Stimulation phase
        if self.biphasicButton.isChecked():
            self.params['stimPhase'] = 'Biphasic'
        elif self.positiveButton.isChecked():
            self.params['stimPhase'] = 'Positive'
        elif self.negativeButton.isChecked():
            self.params['stimPhase'] = 'Negative'


        # Stimulation amplitude
        self.params['stimAmplitude'] = []
        self.params['stimAmplitude'].append(self.stimCh1.value())
        self.params['stimAmplitude'].append(self.stimCh2.value())
        self.params['stimAmplitude'].append(self.stimCh3.value())
        self.params['stimAmplitude'].append(self.stimCh4.value())
        self.params['stimAmplitude'].append(self.stimCh5.value())
        self.params['stimAmplitude'].append(self.stimCh6.value())
        self.params['stimAmplitude'].append(self.stimCh7.value())
        self.params['stimAmplitude'].append(self.stimCh8.value())


        # Trial sequence
        trialSeq = [self.seq1CheckBox.isChecked(), self.seq2CheckBox.isChecked()<<1, self.seq3CheckBox.isChecked()<<2, self.seq4CheckBox.isChecked()<<3,
        self.seq5CheckBox.isChecked()<<4, self.seq6CheckBox.isChecked()<<5, self.seq7CheckBox.isChecked()<<6, self.seq8CheckBox.isChecked()<<7]
        self.params['trialSeq'] = ((((((trialSeq[0] | trialSeq[1]) | trialSeq[2]) | trialSeq[3]) | trialSeq[4]) | trialSeq[5]) | trialSeq[6]) | trialSeq[7]

        # Trial parameters
        self.params['timeON'] = self.timeON.value()
        self.params['timeOFF'] = self.timeOFF.value()
        self.params['timeRead'] = self.timeRead.value()
        self.params['timeStim'] = self.timeStim.value()
        self.params['tremorThreshold'] = int(self.tremorThreshold.value())
        #self.params['tremorMuscle'] = self.tremorMuscle.value() - 1
        
        self.params['ecrChannel']   = self.ecrChannel.value() -1
        self.params['fcrChannel']   = self.fcrChannel.value() -1
        self.params['tremorMuscle'] = self.params['ecrChannel']  # per analisi offline uso l'ecr
        
        
        self.params['numPulses'] = self.numPulses.value()
        self.params['numTrials'] = self.numTrials.value()
        self.params['gain1'] = int(self.gain1.value()*10) # to get int
        self.params['gain2'] = int(self.gain2.value()*10)
        self.params['timeRMS'] = self.timeRMS.value()

        # EMG PARAMETERS
        if self.fs16000radioButton.isChecked():
            self.params['fs'] = 16000
        elif self.fs2000radioButton.isChecked():
            self.params['fs'] = 2000
        elif self.fs4000radioButton.isChecked():
            self.params['fs'] = 4000
        elif self.fs8000radioButton.isChecked():
            self.params['fs'] = 8000

        
        if self.ch2radioButton.isChecked():
            self.params['emgChannels'] = 2
        elif self.ch4radioButton.isChecked():
            self.params['emgChannels'] = 4
        elif self.ch8radioButton.isChecked():
            self.params['emgChannels'] = 8
        elif self.ch16radioButton.isChecked():
            self.params['emgChannels'] = 16

        self.params['recordingTime'] = self.recordingTimeBox.value()
        self.fileName = self.nameEdit.toPlainText()


    def onConnectButtonClicked(self):
        connected = self.fcn_connect_hardware()
        if connected:
            self.connectButton.setStyleSheet("background-color : rgb(9, 249, 17)") # green
            self.disconnectButton.setStyleSheet("background-color : red")
        else:
            self.connectButton.setStyleSheet("background-color : red")

    
    def onDisconnectButtonClicked(self):
        connected = self.fcn_disconnect_hardware()
        if not connected:
            self.disconnectButton.setStyleSheet("background-color : #0d6efd") # blue
            self.connectButton.setStyleSheet("background-color : #0d6efd")
        else:
            self.disconnectButton.setStyleSheet("background-color : orange")
            self.connectButton.setStyleSheet("background-color : #0d6efd")




    def onSaveDirectoryButtonClicked(self):
        self.saveDirectory = str(QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Directory to Save Data')) # choose directory from dialog
        self.statusbar.showMessage(('Saving files in ' + self.saveDirectory)) # print directory in status bar


    
    def fcn_connect_hardware(self):
        """ Function opens serial port to communicate with EAST

        ### Returns
        connected: boolean value 1 = connected 0 = unable to connect
        """
        # Default parameters
        plotTime = 1
        fs = 2000
        numCh = 16

        numChTot = numCh + 1 # EMG + trigger channel
        bufSize = fs*plotTime*2*numChTot

        # !!! Set the correct COM number !!!
        serialCOM = str(self.comBox.currentText()) # !!! ATTENTION: Set the proper COM !!!

        self.serialPort = serial.Serial()
        self.serialPort.baudrate = 921600
        self.serialPort.port = serialCOM
        self.serialPort.timeout = 12
        try:
            self.serialPort.open()
            connected = 1
            self.serialPort.set_buffer_size(rx_size=bufSize)
            #self.serialPort.reset_input_buffer()  # ← AGGIUNGI QUESTA
            self.stop = 0
        except:
            print('Not able to connect to EAST!')
            connected = 0
        
        return connected


    def fcn_disconnect_hardware(self):
        try:
            self.serialPort.close()
            connected = 0
        except:
            connected = 1
        return connected


    
    def fcn_start_EMG(self):
        commnadSize = 25
        mode = 0 # Working mode: 0 = EMG, 1 = Test Mode, 2 = Impedance check
        self.stop = 0
        stimFreqDataSet = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100])
        stimDurDataSet = np.array([100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 750, 800, 850, 900, 950, 1000])
        stimAmpDataSet = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

        confString = []

        # BYTE 1
        confString.append(commnadSize)

        # BYTE 2
        # Mode
        if mode == 0:
            confString.append(int('00000000',2))
        elif mode == 1:
            confString.append(int('01000000',2))
        elif mode == 2:
            confString.append(int('10000000',2))
        else:
            confString.append(int('00000000',2))

        # Sampling frequency MIRAR ESTO!!!!!!!!!!!!!!!
        if self.params['fs'] == 16000:
            confString[-1] = confString[-1] + int('00110000',2)
        elif self.params['fs'] == 2000:
            confString[-1] = confString[-1] + int('00000000',2)
        elif self.params['fs'] == 4000:
            confString[-1] = confString[-1] + int('00010000',2)
        elif self.params['fs'] == 8000:
            confString[-1] = confString[-1] + int('00100000',2)


        # Channels
        if self.params['emgChannels'] == 2:
            confString[-1] = confString[-1] + int('00000000',2)
        elif self.params['emgChannels'] == 4:
            confString[-1] = confString[-1] + int('00000100',2)
        elif self.params['emgChannels'] == 8:
            confString[-1] = confString[-1] + int('00001000',2)
        elif self.params['emgChannels'] == 16:
            confString[-1] = confString[-1] + int('00001100',2)

        # Start data trasnfer by setting GO bit = 1
        confString[-1] = confString[-1] +1

        # BYTE 3. Stimulation active channels (all channels off)
        confString.append(0)

        # BYTE 4. Stimulation frequency and type
        stimFreqIndex = np.argmin(abs(stimFreqDataSet - self.params['stimFrequency'])) + 1 # to keep MATLAB indexing +1 
        
        if self.params['stimPhase'] == 'Biphasic':
            stimType = 3*32
        elif self.params['stimPhase'] == 'Positive':
            stimType = 1*32
        elif self.params['stimPhase'] == 'Negative':
            stimType = 2*32

        confString.append(stimType + stimFreqIndex)

        # BYTE 5. Pulse width
        stimDurIndex = np.argmin(abs(stimDurDataSet - self.params['pulseWidth'])) + 1 # to keep MATLAB indexing +1 
        confString.append(stimDurIndex)

        # BYTES 6 -13. Stimulation amplitude
        for i in range(8):
            confString.append(0)

        confString.append(self.params['timeON'])
        confString.append(self.params['timeOFF'])
        confString.append(self.params['timeRead'])
        confString.append(self.params['timeStim'])
        confString.append(self.params['trialSeq'])

        if self.params['mode'] == 'Manual':
            confString.append(0)
        elif self.params['mode'] == 'Out-of-phase':
            confString.append(1)
        elif self.params['mode'] == 'SATS':
            confString.append(2)
        elif self.params['mode'] == 'CON':
            confString.append(3)

        confString.append(self.params['tremorThreshold'])
        confString.append(self.params['gain1'])
        confString.append(self.params['gain2'])
        confString.append(self.params['timeRMS'])


         # confString.append(self.params['tremorMuscle'])
        confString.append((self.params['ecrChannel'] << 4) | self.params['fcrChannel'])




        # CRC8
        confString.append(self.CRC8(confString, commnadSize-1))

        sendConfstring = [int(x) for x in confString]

        # Write command string in uint8
        self.serialPort.write(struct.pack("{0}B".format(len(sendConfstring)),*sendConfstring))

        self.readEMG()

        commandSent = 1
        self.stop = 0

        return commandSent


    def readEMG(self, display_channels=None):
        self.stop = 0
        numChannelsTot = self.params['emgChannels'] + 1 # for trigger channel
        #numChannelsTot = self.params['emgChannels']
        tRead = self.params['recordingTime'] # in seconds
        first = 1

        # if numCyc > 1:
        #     numCyc += 1

        
        plotTime = 1
        # offset = 2
        """
        if str(self.comBox.currentText()) == 'COM7':
            offset = [4.39, 4.39, 0, -480] # EAST viejo COM7
        elif str(self.comBox.currentText()) == 'COM6':
            offset = [-0.232, -0.047, 0, -480] # EAST nuevo COM6
        elif str(self.comBox.currentText()) == 'COM5':
            # offset = [-2923+140+2787.7-0.2149, -110.313+120-5.5055-7.8652+4.9795+3.1155, -150, -480] # EAST más nuevo tornillo plata
            offset = [4.4851, 4.4113, 0, -480] # EAST más nuevo tornillo plata COM5
        elif str(self.comBox.currentText()) == 'COM3':
            offset = [4.4851, 4.4113, 0, -480] # EAST 3
        else:
            offset = [-0.232, -0.047, 0, -480] # EAST nuevo COM6
        """
        # modifica bracelet : mettere tutti gli offset a 0
        offset=[0]*numChannelsTot;


        # convFactor = 0.000286 # in mV
        convFactor = 0.000286*1000 # in mV
        # bytesExpected = 2*numChannelsTot*self.params['fs']
        # nSamples = self.params['fs'] 
        nSamples = 125  # Receive 125 samples per channel -> plot every 0.0625 seconds (125* (1/2000)) -> 16 Hz
        bytesExpected = 2*numChannelsTot*nSamples 
        bytesRead = 0
        currentBytesData = []
        self.emgRaw = []

        numCyc = tRead * self.params['fs'] / nSamples
        currentCyc = 0

        if numCyc == 0:
            numCyc = 99999

        numSamples = self.params['fs']*plotTime
        t = np.linspace(0, plotTime, numSamples)

        startTime = time.perf_counter()
        previousPlottingTimer = time.perf_counter()
        order = 200
        wn = 10.0 / (self.params['fs'] / 2.0)
        fir_bLP = signal.firwin(order + 1, wn, window='hamming', pass_zero='lowpass')
        self.serialPort.reset_input_buffer()  # ← AGGIUNGI QUESTA
        print(f"[START readEMG] byte nel buffer: {self.serialPort.in_waiting}")
        print(f"  bytesExpected: {bytesExpected}  ({numChannelsTot} ch x {nSamples} campioni x 2 byte)")





        # while time.perf_counter() - startTime <= numCyc:
        while currentCyc < numCyc:
            plottingTimer = time.perf_counter()
            self.getParameters()
            y_plot = []
            y_add = []


            if self.stop == 1:
                break


            # if plottingTimer - previousPlottingTimer >= plotTime:
            if self.serialPort.in_waiting >= bytesExpected:
                currentCyc += 1
                previousPlottingTimer = time.perf_counter()
                if self.stop == 0:
                    currentBytesData = []
                    chunk = self.serialPort.read(bytesExpected) # serial port reads all you ask until timeout
                    if chunk == b'':
                        raise RuntimeError('Serial connection broken')

                    currentBytesData.append(chunk)
                    bytesRead += len(chunk)
                    currentBytesDataJoined = b''.join(currentBytesData)
                    #qui avviene il rashape
                    currentData = np.array(struct.unpack("{0}h".format(int(len(currentBytesDataJoined)/2)),currentBytesDataJoined), dtype=np.float64).reshape((numChannelsTot,nSamples), order="F") # size nChnnels rows
                    if type(self.emgRaw) == list:
                        self.emgRaw = currentData
                    else:
                        self.emgRaw = np.append(self.emgRaw, currentData, axis=1) # Grow horizontally

                    # modifica bracelet
                    channels_to_show = display_channels if display_channels is not None \
                            else list(range(self.params['emgChannels']))
                    for i in channels_to_show:
                        y_plot.append(currentData[i,:]*convFactor - offset[i])

                    # calcolo envelope per i canali mostrati
                    y_add = []
                    for i in channels_to_show:
                        env = signal.lfilter(fir_bLP, 1, np.abs(currentData[i,:]*convFactor))
                        y_add.append(env)
                        
                        #for i in range(self.params['emgChannels']):
                        #    y_plot.append(currentData[i,:]*convFactor - offset[i])
                        
                        
                        
                        """if self.params['emgChannels'] < 3:
                            # emgEnvelope, burstDetected = self.fcn_tremor_detection(currentData[i,:]*convFactor + offset*(i+1), self.params['fs'], 40, 1)
                            # emgEnvelope, burstDetected = self.fcn_tremor_detection(currentData[i,:]*convFactor - offset[i], self.params['fs'], 40, 1)
                            

                            if self.params['emgChannels'] == 2:
                                # plot data
                                # y_plot.append(currentData[i,:]*convFactor + offset*(i+1))
                                y_plot.append(currentData[i,:]*convFactor - offset[i])
                                # calculate envelope of the last 5 seconds
                                emgEnvelope = self.calculate_envelope(self.emgRaw[i,:], self.params['fs'])
                                # if self.emgRaw.shape[1] < 5 * self.params['fs']: # if there are less than second of samples
                                #     emgEnvelope = self.calculate_envelope(self.emgRaw[i,:], self.params['fs']) # calculate envelope with what you have
                                # else:
                                #     emgEnvelope = self.calculate_envelope(self.emgRaw[i,-5 * self.params['fs']:], self.params['fs']) # calculate envelope with the last second
                                # y_add.append(emgEnvelope[-y_plot[0].shape[0]:]) # only keep the last samples
                                y_add.append(emgEnvelope) # only keep the last samples
                            else:
                                # plot data
                                # y_plot.append(currentData[i,:]*convFactor + offset*(i+1))
                                y_plot.append(currentData[i,:]*convFactor - offset[i])

                            if first < self.params['emgChannels']+1:
                                first += 1
                        else:
                            # plot data
                            # y_plot.append(currentData[i,:]*convFactor + offset*(i+1))
                            y_plot.append(currentData[i,:]*convFactor - offset[i])
                            if first < self.params['emgChannels']+1:
                                first += 1
                        """
                        # Refresh state of application
                    QtWidgets.QApplication.processEvents()
                    self.extraWindow.updateSeparatePlot(x_to_plot=t, y_to_plot=y_plot, additional_x=t, additional_y=[])
                    #self.extraWindow.updateSeparatePlot(x_to_plot=t, y_to_plot=y_plot, additional_x=t, additional_y=y_add)


        for j in range(numChannelsTot):
            self.emgRaw[j,:] = self.emgRaw[j,:]*convFactor - offset[j]

        commandSent = self.fcn_stop_stim()
        #SAVE DATA
        if self.saveCheckBox.isChecked() == False: # to avoid losing data that needs to be saved
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setWindowTitle('EAST GUI')
            msg.setText('Do you want to save the recorded data?')
            msg.setInformativeText('It will saved in ' + self.saveDirectory)
            msg.setStandardButtons(QtWidgets.QMessageBox.Save|QtWidgets.QMessageBox.Discard)
            msg.setDefaultButton(QtWidgets.QMessageBox.Save)

            x = msg.exec_()

            if x == QtWidgets.QMessageBox.Save:
                self.saveFile()
            else:
                self.statusbar.showMessage('Data discarded')

        else:
            self.saveFile()


    def saveFile(self):
        try:
            checkFile = os.path.join(self.saveDirectory, self.fileName + '.mat')
            i = 1
            while os.path.isfile(checkFile): # if file already exists, avoid overwriting by changing name
                newFileName = self.fileName + str(i)
                checkFile = os.path.join(self.saveDirectory, newFileName + '.mat')
                i += 1
            
            # SAVE DATA HERE
            dataDictionary = {'emgRaw': self.emgRaw, 'fs_EMG': self.params['fs'], 'params': self.params}
            savemat(checkFile, dataDictionary)
            self.statusbar.showMessage('Saved!')
        except:
            self.statusbar.showMessage('Something went wrong')

    
    def calculate_envelope(self, emg, fs):
        [bBP,aBP] = signal.butter(2, [10/(fs/2), 400/(fs/2)], 'band')
        [bLP,aLP] = signal.butter(4, 10/(fs/2), 'low')
        emg = emg.T
        emgFil = signal.filtfilt(bBP, aBP, emg, padtype = 'odd', padlen=3*(max(len(bBP),len(aBP))-1), axis=0)
        emgEnvelope = signal.filtfilt(bLP,aLP, abs(emgFil), padtype = 'odd', padlen=3*(max(len(bLP),len(aLP))-1), axis=0)
        if emgEnvelope.size == 0:
            emgEnvelope = np.zeros((1000,2))

        emgEnvelope = emgEnvelope.T
        return emgEnvelope



    def fcn_tremor_detection(self, emg, fs, tremorThreshold, nMuscle):
        [bBP,aBP] = signal.butter(2, [10/(fs/2), 400/(fs/2)], 'band')
        [bLP,aLP] = signal.butter(4, 10/(fs/2), 'low')

        emg = emg.T
        trialLength = max(emg.shape)
        windowStim = self.params['timeStim']

        emgFil = signal.filtfilt(bBP, aBP, emg, padtype = 'odd', padlen=3*(max(len(bBP),len(aBP))-1), axis=0)
        emgEnvelope = signal.filtfilt(bLP,aLP, abs(emgFil), padtype = 'odd', padlen=3*(max(len(bLP),len(aLP))-1), axis=0)

        burstDetected = np.zeros((max(emg.shape),))


        tremorBurstLoc = []
        for j in range(1, max(emgEnvelope.shape)-1):
            if emgEnvelope[j] > tremorThreshold and emgEnvelope[j] < 200 and emgEnvelope[j] > emgEnvelope[j+1] and emgEnvelope[j] > emgEnvelope[j-1]:
                tremorBurstLoc.append(j)

        nBursts = len(tremorBurstLoc)
        ibi = []

        if nBursts > 2:
            for i in range(nBursts-1):
                ibi.append(abs(tremorBurstLoc[i] - tremorBurstLoc[i+1]))

            ibiMean = mean(ibi)/fs
            nextBursts = []
            k = 1
            while round(tremorBurstLoc[-1]/fs + ibiMean*k*fs) < windowStim: # mirar lo de /fs
                nextBursts.append(round(tremorBurstLoc[-1] + ibiMean*k*fs))
                k += 1

        burstDetectedDummy = np.zeros((max(emg.shape)))
        burstDetectedDummy[tremorBurstLoc] = 0.05

        if emgEnvelope.size == 0:
            emgEnvelope = np.zeros((1000,1))

        burstDetected[:] = burstDetectedDummy

        emgEnvelope = emgEnvelope.T
        return emgEnvelope, burstDetected



    def fcn_stop_stim(self):
        commnadSize = 25
        mode = 0 # Working mode: 0 = EMG, 1 = Test Mode, 2 = Impedance check
        self.stop = 0
        stimFreqDataSet = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100])
        stimDurDataSet = np.array([100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 750, 800, 850, 900, 950, 1000])
        stimAmpDataSet = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

        confString = []

        # BYTE 1
        confString.append(commnadSize)

        confString.append(0)

        confString.append(0)

        # BYTE 4. Stimulation frequency and type
        stimFreqIndex = np.argmin(abs(stimFreqDataSet - self.params['stimFrequency'])) + 1 # to keep MATLAB indexing +1 
        

        confString.append(32 + stimFreqIndex)

        # BYTE 5. Pulse width
        stimDurIndex = np.argmin(abs(stimDurDataSet - self.params['pulseWidth'])) + 1 # to keep MATLAB indexing +1 
        confString.append(stimDurIndex)

        # BYTES 6 -13. Stimulation amplitude
        for i in range(8):
            confString.append(0)

        confString.append(0)
        confString.append(0)
        confString.append(0)
        confString.append(0)
        confString.append(0)
        confString.append(0)
        confString.append(0)
        confString.append(0)
        confString.append(0)
        confString.append(0)
        confString.append(0)

        # CRC8
        confString.append(self.CRC8(confString, commnadSize-1))

        # Write command string in uint8
        self.serialPort.write(struct.pack("{0}B".format(len(confString)),*confString))

        commandSent = 1
        self.stop = 1

        return commandSent
              
                
    def fcn_start_stim(self):
        commnadSize = 25
        mode = 0 # Working mode: 0 = EMG, 1 = Test Mode, 2 = Impedance check
        self.stop = 0
        stimFreqDataSet = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100])
        stimDurDataSet = np.array([100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 750, 800, 850, 900, 950, 1000])
        stimAmpDataSet = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

        confString = []

        # BYTE 1
        confString.append(commnadSize)

        # BYTE 2
        # Mode
        if mode == 0:
            confString.append(int('00000000',2))
        elif mode == 1:
            confString.append(int('01000000',2))
        elif mode == 2:
            confString.append(int('10000000',2))
        else:
            confString.append(int('00000000',2))

        # Sampling frequency MIRAR ESTO!!!!!!!!!!!!!!!
        if self.params['fs'] == 16000:
            confString[-1] = confString[-1] + int('00110000',2)
        elif self.params['fs'] == 2000:
            confString[-1] = confString[-1] + int('00000000',2)
        elif self.params['fs'] == 4000:
            confString[-1] = confString[-1] + int('00010000',2)
        elif self.params['fs'] == 8000:
            confString[-1] = confString[-1] + int('00100000',2)


        # Channels
        if self.params['emgChannels'] == 2:
            confString[-1] = confString[-1] + int('00000000',2)
        elif self.params['emgChannels'] == 4:
            confString[-1] = confString[-1] + int('00000100',2)
        elif self.params['emgChannels'] == 8:
            confString[-1] = confString[-1] + int('00001000',2)
        elif self.params['emgChannels'] == 16:
            confString[-1] = confString[-1] + int('00001100',2)

        # Start data trasnfer by setting GO bit = 1
        confString[-1] = confString[-1] +1

        # BYTE 3. Stimulation active channels 
        confString.append(0)
        for i in range(8):
            if self.params['stimAmplitude'][i] > 0:
                confString[-1] = confString[-1] + (1<<i)
            else:
                confString[-1] = confString[-1] + (0<<i)

        # BYTE 4. Stimulation frequency and type
        stimFreqIndex = np.argmin(abs(stimFreqDataSet - self.params['stimFrequency'])) + 1 # to keep MATLAB indexing +1 
        
        if self.params['stimPhase'] == 'Biphasic':
            stimType = 3*32
        elif self.params['stimPhase'] == 'Positive':
            stimType = 1*32
        elif self.params['stimPhase'] == 'Negative':
            stimType = 2*32

        confString.append(stimType + stimFreqIndex)

        # BYTE 5. Pulse width
        stimDurIndex = np.argmin(abs(stimDurDataSet - self.params['pulseWidth'])) + 1 # to keep MATLAB indexing +1 
        confString.append(stimDurIndex)

        # BYTES 6 -13. Stimulation amplitude
        for i in range(8):
            stimAmpIndex = np.argmin(abs(stimAmpDataSet - self.params['stimAmplitude'][i])) + 1 # to keep MATLAB indexing +1 
            confString.append(stimAmpIndex)

        confString.append(self.params['timeON'])
        confString.append(self.params['timeOFF'])
        confString.append(self.params['timeRead'])
        confString.append(self.params['timeStim'])
        confString.append(self.params['trialSeq'])

        if self.params['mode'] == 'Manual':
            confString.append(0)
        elif self.params['mode'] == 'Out-of-phase':
            confString.append(1)
        elif self.params['mode'] == 'SATS':
            confString.append(2)
        elif self.params['mode'] == 'CON':
            confString.append(3)

        confString.append(self.params['tremorThreshold'])
        confString.append(self.params['gain1'])
        confString.append(self.params['gain2'])
        confString.append(self.params['timeRMS'])
        
        
        # confString.append(self.params['tremorMuscle'])
        confString.append((self.params['ecrChannel'] << 4) | self.params['fcrChannel'])



        # CRC8
        confString.append(self.CRC8(confString, commnadSize-1))

        # Write command string in uint8
        self.serialPort.write(struct.pack("{0}B".format(len(confString)),*confString))

        self.readEMG(display_channels=[self.params['ecrChannel'], self.params['fcrChannel']])

        commandSent = 1
        self.stop = 0

        return commandSent


    def CRC8(self, vector, lenVector):
        crc = 0
        j = 0

        while lenVector > 0:
            Extract = vector[j]

            for i in range(8,0,-1):
                if Extract%1 == 0: # int
                    dummy = int(Extract)%2
                else: # float
                    dummy = 1
                Sum = operator.xor(crc%2, dummy) # % modulo operation
                crc = crc//2 # floor division

                if Sum > 0 :
                    st = np.zeros((8))
                    a = bin(crc)[2:].zfill(8)
                    b = bin(140)[2:].zfill(8)

                    for k in range(8):
                        st[k] = not((a[k] == b[k]))

                    crc = int(np.array2string(st).replace('.','').replace(' ','').replace('[','').replace(']',''),2)

                Extract = Extract//2

            lenVector = lenVector - 1
            j = j + 1

        return crc


    def onAnalyzeButtonClicked(self):
        fileFilter = 'Data File (*.mat)'
        self.dataFilePath = str(QtWidgets.QFileDialog.getOpenFileName(self, 'Select File to Analyze', directory=self.saveDirectory, filter=fileFilter)[0]) # choose .mat file from dialog
        # dialog returns a tuple. First element is the path, second element is the filter
        self.statusbar.showMessage(('Loading ' + self.dataFilePath))

        # Load file
        emgData = loadmat(self.dataFilePath)
        self.mainSATS(emgData)

    
    def mainSATS(self, emgData):
        # Close open plot windows
        try:
            self.plotWindow.close()
        except:
            pass
        self.getParameters()
        fs = float(emgData['fs_EMG'][0][0])
        # Filters
        [bBP,aBP] = signal.butter(2, [10/(fs/2), 400/(fs/2)], 'band')
        [bLP,aLP] = signal.butter(4, 10/(fs/2), 'low')
        # FIR filter
        order = 200
        lpCutoff = 10
        wn = (lpCutoff)/(fs/2)
        fir_bLP = signal.firwin(order+1, wn, window = 'hamming', pass_zero = 'lowpass')

        emgRaw = emgData['emgRaw']
        emgRaw = emgRaw.T # now shape n samples x n muscles
        ecr = self.params['ecrChannel']   # già 0-based
        fcr = self.params['fcrChannel']   # già 0-based
        emgRaw = emgRaw[:, [ecr, fcr]]   # riduce a [N × 2] → il resto del codice non cambia

        trialLength = max(emgRaw.shape)
        windowRecON = np.zeros([trialLength,]) # when system is recording

        # Parameters
        recWindow = int(round(self.params['timeRead'] * fs)) # windows in samples
        stimWindow = int(round(self.params['timeStim'] * fs))
        
        
        #muscle = self.params['tremorMuscle']  # di default è l'ecr 
        muscle=0; # dopo lo slicing di emgRaw ai solo due canali di interesse, ecr è sempre in posizione 0 .
        
        tremorThreshold = self.params['tremorThreshold'] # in mV
        emgRMSThreshold = np.array([0.004, 0.002]) # Placeholder until RMS threshold is calculated
        currentRMS = np.array([0.004, 0.002]) # Placeholder until RMS is calculated
        stimDuration = 0.25 # stim tremor duty cycle (0-1)
        windowRMS = int(round(self.params['timeRMS']/1000 * fs)) # in seconds * fs = samples
        gain1 = self.params['gain1']/10 # gain is saved multiplied by 10 to get an integer and send it as uint8 in the confString
        gain2 = self.params['gain2']/10

        state = 0
        stimulationOutput = np.zeros([trialLength,2])
        burstDetected = []
        burstPredicted = []
        currentPosition = 0
        plotMax = 250

        w = int(1 * fs)
        k = 0

        while w < trialLength:
            # SWITCH
            if state == 0: # init
                state = 1
            
            elif state == 1: # tremor detection
                windowRecON[w-1:w+recWindow] = plotMax
                currentEMG = emgRaw[w-1:w+recWindow, :]

                # Envelope
                emgEnvelope = signal.lfilter(fir_bLP, 1, np.abs(currentEMG), axis=0) # filter each column (along all the samples for each muscle)

                # Update RMS threshold
                emgRMSThreshold[0] = np.sqrt(np.mean(currentEMG[:,0]**2)) * gain1
                emgRMSThreshold[1] = np.sqrt(np.mean(currentEMG[:,1]**2)) * gain2

                # Find tremor bursts
                tremorBurst = np.zeros([len(emgEnvelope),])
                tremorBurstLoc = []

                for j in range(1,len(emgEnvelope)-1):
                    # Above defined tremor threshold, below noise level and a local maxima
                    if emgEnvelope[j,muscle] > tremorThreshold and emgEnvelope[j,muscle] < 300 and emgEnvelope[j,muscle] > emgEnvelope[j+1,muscle] and emgEnvelope[j,muscle] > emgEnvelope[j-1,muscle]:
                        tremorBurstLoc.append(round(j-order/2))

                tremorBurst[[round(x) for x in tremorBurstLoc]] = 30
                nBursts = len(tremorBurstLoc)

                # Calculate interburst interval
                ibi = []
                if nBursts > 4:
                    tremorPresent = 1
                    state = 2
                    for i in range(nBursts-1):
                        ibi.append(abs(tremorBurstLoc[i] - tremorBurstLoc[i+1]))

                    ibiMean = mean(ibi)/fs # in seconds
                    freqTremor = 1/ibiMean
                    nextBurst = []
                    j = 1
                    while round(tremorBurstLoc[-1] + ibiMean*j*fs) < stimWindow:
                        nextBurst.append(round(tremorBurstLoc[-1] + ibiMean*j*fs))
                        j +=1

                    burstDetected.extend([x+w for x in tremorBurstLoc])
                    burstPredicted.extend([x+w for x in nextBurst])
                else:
                    tremorPresent = 0

                    # if w + recWindow + stimWindow*2 < trialLength:
                    #     w = w + recWindow + stimWindow # MIRAR!
                    # else:
                    #     w = trialLength +1
                    # Cambiado
                    w = w + recWindow 

                    state = 1

                
            elif state == 2:
                if k < trialLength:
                    w = w + recWindow
                    k = w

                    while k < w + stimWindow - round(windowRMS):
                        currentEMG = emgRaw[k-1:round(k+windowRMS),:]
                        currentRMS[0] = np.sqrt(np.mean(currentEMG[:,0]**2))
                        currentRMS[1] = np.sqrt(np.mean(currentEMG[:,1]**2))
                        currentPosition = round(k+windowRMS)

                        if currentRMS[1] > emgRMSThreshold[1]:
                            stimulationOutput[currentPosition-1:round(currentPosition+fs*ibiMean*stimDuration),0] = plotMax

                        if currentRMS[0] > emgRMSThreshold[0]:
                            stimulationOutput[currentPosition-1:round(currentPosition+fs*ibiMean*stimDuration),1] = -plotMax

                        if currentRMS[0] > emgRMSThreshold[0] or currentRMS[1] > emgRMSThreshold[1]:
                            k = round(currentPosition + fs*ibiMean*stimDuration)
                        else:
                            k = currentPosition + 1

                    
                    # if w + stimWindow*2 < trialLength:
                    #     w = w + stimWindow
                    #     state = 1
                    # else:
                    #     w = trialLength + 1
                    #     state = 0

                    w = w + stimWindow
                    state = 1


                else:
                    w = trialLength + 1
                    state = 0

        # Filtering 
        emgFilEnvelope = signal.lfilter(fir_bLP, 1, np.abs(emgRaw), axis=0)
        emgFilEnvelope1 = np.flip(signal.lfilter(fir_bLP, 1, np.abs(np.flip(emgRaw,0)), axis=0),0) # flip columns (last row is the first row)
        t = np.arange(0,(len(emgFilEnvelope))/fs,1/fs)

        burstDetectedSignal = np.zeros([len(emgRaw),])
        burstPredictedSignal = np.zeros([len(emgRaw),])
        burstDetectedSignal[burstDetected] = plotMax
        burstPredictedSignal[[x for x in burstPredicted if x < len(emgRaw)]] = plotMax

        # Plotting
        self.plotWindow = MatplotlibWindow()
        self.plotWindow.show()
        self.plotWindow.plot(t, emgRaw, emgFilEnvelope, burstDetectedSignal, burstPredictedSignal, windowRecON, stimulationOutput, order)



    
    def closeEvent(self, event):
		# Close open windows
        try:
            self.extraWindow.close()
        except:
            pass

        # Close open plot windows
        try:
            self.plotWindow.close()
        except:
            pass

        event.accept()


class MatplotlibWindow(QtWidgets.QWidget):
    """ This opens a new window to display another graph.
    This "window" is a QWidget. If it has no parent, it will appear as a free-floating window.
    """

    def __init__(self):
        super().__init__()
        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that
        # displays the 'figure'it takes the
        # 'figure' instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
  
        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # creating a Vertical Box layout
        layout = QtWidgets.QVBoxLayout()

        # adding tool bar to the layout
        layout.addWidget(self.toolbar)
          
        # adding canvas to the layout
        layout.addWidget(self.canvas)

        # set zero margins
        layout.setContentsMargins(0, 0, 0, 0)
        self.setStyleSheet('background-color: white')

        self.setWindowIcon(QtGui.QIcon(':/icons/Icons/lightning-bolt.png'))
        self.setWindowTitle('Plot')

        # setting layout to the main window
        self.setLayout(layout)



    def plot(self, t, emgRaw, emgFilEnvelope, burstDetectedSignal, burstPredictedSignal, windowRecON, stimulationOutput, order):  
        # clearing old figure
        self.figure.clear()
  
        plt.plot(t, emgRaw[:,0], label='ECR')
        plt.plot(t, emgRaw[:,1], label = 'FCR')
        plt.plot(t[0:round(len(t)-order/2+1)], emgFilEnvelope[round(order/2)-1:,0], label='ECR envelope')
        plt.plot(t[0:round(len(t)-order/2+1)], emgFilEnvelope[round(order/2)-1:,1], label='FCR envelope')
        plt.plot(t,burstDetectedSignal, label='Tremor detected')
        plt.plot(t,burstPredictedSignal, label='Tremor predicted')
        plt.plot(t,stimulationOutput[:,0], label='Stimulation ECR',linestyle='-.')
        plt.plot(t,stimulationOutput[:,1], label='Stimulation FCR',linestyle='-.')
        plt.plot(t,windowRecON, label='Recording window',linestyle='--')

        plt.legend()

        plt.xlabel('s')
        plt.ylabel('mV')
        plt.subplots_adjust(top=0.981,bottom=0.075,left=0.057,right=0.99,hspace=0.2,wspace=0.2)
  
        # refresh canvas
        self.canvas.draw()




# Vecchio another window 
"""
class AnotherWindow(QtWidgets.QWidget):
    This opens a new window to display another graph.
    This "window" is a QWidget. If it has no parent, it will appear as a free-floating window.
    

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QHBoxLayout()
        layout.setObjectName('layout')


        self.continuousPlot = QtWidgets.QWidget()
        self.continuousPlot.setObjectName('continuousPlot')
        layout3 = QtWidgets.QVBoxLayout(self.continuousPlot)
        layout3.setObjectName('layout3')
        layout3.setContentsMargins(0, 0, 0, 0)
        self.continuousGraphWidget = pg.PlotWidget()
        # Properties 
        self.continuousGraphWidget.setBackground('w')
        pen = pg.mkPen(color=(0, 0, 255), width=1.5)
        self.continuousGraphWidget.setLabel('left', 'mV')
        self.continuousGraphWidget.setLabel('bottom', 's')
        layout3.addWidget(self.continuousGraphWidget)

        self.continuousGraphWidget2 = pg.PlotWidget()
        # Properties 
        self.continuousGraphWidget2.setBackground('w')
        pen = pg.mkPen(color=(0, 0, 255), width=1.5)
        pen2 = pg.mkPen(color=(242, 90, 44), width=2)
        self.continuousGraphWidget2.setLabel('left', 'mV')
        self.continuousGraphWidget2.setLabel('bottom', 's')
        layout3.addWidget(self.continuousGraphWidget2)

        layout.addWidget(self.continuousPlot)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.setStyleSheet('background-color: white')

        self.setWindowIcon(QtGui.QIcon(':/icons/Icons/lightning-bolt.png'))
        self.setWindowTitle('Plot')
        w = 1000
        h = 800
        self.setGeometry(100,100,w,h)

        # Initialize line
        self.x = list(range(100))  # 100 time points
        self.y = np.zeros((100,))  # 100 data points
        self.y2 = np.zeros((100,))  # 100 data points
        self.x_add = np.zeros((100,))  # 100 data points
        self.y_add = np.zeros((100,))  # 100 data points
        self.y_add2 = np.zeros((100,))  # 100 data points
        self.data_line = self.continuousGraphWidget.plot(self.x, self.y, pen=pen)
        self.data_line2 = self.continuousGraphWidget2.plot(self.x, self.y2, pen=pen)
        self.data_add = self.continuousGraphWidget.plot(self.x_add, self.y_add, pen=pen2)
        self.data_add2 = self.continuousGraphWidget2.plot(self.x_add, self.y_add2, pen=pen2)
        # SETUP VARIABLES 
        self.n_plot = 0 # Points available to continuously plot

    
    def updateSeparatePlot(self, x_to_plot, y_to_plot, additional_x, additional_y):

        if self.n_plot == 0:
            self.y = y_to_plot[0]
            self.y2 = y_to_plot[1]
            self.x = x_to_plot
            if len(additional_y) > 0:
                self.y_add = additional_y[0]
                self.y_add2 = additional_y[1]
                self.x_add = additional_x
        else:
            self.y = np.append(self.y,y_to_plot[0])
            self.y2 = np.append(self.y2,y_to_plot[1])
            if len(additional_y) > 0:
                # self.y_add = np.append(self.y_add,additional_y[0])
                # self.y_add2 = np.append(self.y_add2,additional_y[1])
                self.y_add = additional_y[0]
                self.y_add2 = additional_y[1]

        self.n_plot = len(self.y)
        plot_len = len(self.x)
        plot_len_add = len(self.x_add)

        if self.n_plot < plot_len:
            self.data_line.setData(self.x[0:self.n_plot], self.y[0:self.n_plot])
            self.data_line2.setData(self.x[0:self.n_plot], self.y2[0:self.n_plot])
            self.data_add.setData(self.x_add[0:self.n_plot], self.y_add[0:self.n_plot])
            self.data_add2.setData(self.x_add[0:self.n_plot], self.y_add2[0:self.n_plot])
        else:
            self.data_line.setData(self.x, self.y[self.n_plot-plot_len:])
            self.data_line2.setData(self.x, self.y2[self.n_plot-plot_len:])
            self.data_add.setData(self.x_add, self.y_add[self.n_plot-plot_len_add:])
            self.data_add2.setData(self.x_add, self.y_add2[self.n_plot-plot_len_add:])

        self.continuousGraphWidget.setYRange(-200, 200, padding=0)
        self.continuousGraphWidget2.setYRange(-200, 200, padding=0)


        



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    app.exec_()
    """

# nuovo anoter window 
class AnotherWindow(QtWidgets.QWidget):
    """ Finestra di plot real-time.

    ── MODIFICA BRACELET ──────────────────────────────────────────────────────
    Accetta il parametro ``num_ch`` (default=2) per creare una griglia di
    subplot, uno per ciascun canale EMG del bracelet (fino a 16).
    Con 2 canali il comportamento è identico all'originale.
    Con più canali i subplot sono disposti in una griglia PLOT_COLS colonne
    dentro una QScrollArea, cosicché tutti i tracciati siano sempre visibili.
    ──────────────────────────────────────────────────────────────────────────
    """

    def __init__(self, num_ch=2):
        super().__init__()
        self.num_ch = num_ch

        # ── Layout principale ─────────────────────────────────────────────────
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        # ── ScrollArea: contiene la griglia di plot ───────────────────────────
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

        self.plots      = []   # PlotWidget per ogni canale
        self.data_lines = []   # curva EMG raw
        self.data_adds  = []   # curva inviluppo (solo per ≤2 canali)

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

            line     = pw.plot([], [], pen=pen_emg)
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

        # Dimensioni finestra: più alta per molti canali
        w = 1000
        h = min(200 * rows, 900)
        self.setGeometry(100, 100, w, h)

        # ── Buffers dati ──────────────────────────────────────────────────────
        self.x      = list(range(100))
        self.x_add  = np.zeros((100,))
        self.y_data     = [np.zeros((100,)) for _ in range(num_ch)]
        self.y_add_data = [np.zeros((100,)) for _ in range(num_ch)]
        self.n_plot = 0   # campioni accumulati finora


    def updateSeparatePlot(self, x_to_plot, y_to_plot, additional_x, additional_y):
        """Aggiorna tutti i subplot con i nuovi campioni.

        y_to_plot  – lista con un array per ogni canale EMG
        additional_y – lista con gli inviluppi (solo nel caso a 2 canali,
                       altrimenti lista vuota)
        """
        if self.n_plot == 0:
            # Prima chiamata: inizializza i buffer
            for i in range(self.num_ch):
                self.y_data[i] = y_to_plot[i]
            self.x     = x_to_plot
            self.x_add = additional_x
            for i in range(len(additional_y)):
                self.y_add_data[i] = additional_y[i]
        else:
            # Chiamate successive: accoda i nuovi campioni
            for i in range(self.num_ch):
                self.y_data[i] = np.append(self.y_data[i], y_to_plot[i])
            # L'inviluppo viene sostituito (non accodato) come nell'originale
            for i in range(len(additional_y)):
                self.y_add_data[i] = additional_y[i]

        self.n_plot      = len(self.y_data[0])
        plot_len         = len(self.x)
        plot_len_add     = len(self.x_add)

        for i in range(self.num_ch):
            if self.n_plot < plot_len:
                self.data_lines[i].setData(
                    self.x[0:self.n_plot],
                    self.y_data[i][0:self.n_plot])
            else:
                self.data_lines[i].setData(
                    self.x,
                    self.y_data[i][self.n_plot - plot_len:])

            # Inviluppo: y_add_data è già finestrato a numSamples, va plottato
            # direttamente senza usare n_plot come offset (evita shape mismatch)
            if i < len(additional_y):
                env = self.y_add_data[i]
                self.data_adds[i].setData(
                    self.x_add[0:len(env)],
                    env)

            # setYRange già impostato in __init__; non serve ripeterlo ogni frame




if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    app.exec_()
