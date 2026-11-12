import time
import struct
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


class CalibrationWorker(QThread):
    # Segnale emesso ad ogni chunk ricevuto (16 canali x 125 campioni)
    signal_chunk = pyqtSignal(np.ndarray)
    signal_error = pyqtSignal(str)

    def __init__(self, ui=None, parent=None):
        super().__init__(parent)
        self.ui = ui
        self.is_running = False

    def stop(self):
        """Richiede l'arresto del loop di acquisizione."""
        self.is_running = False
        self.wait()

    def run(self):
        self.is_running = True
        num_ch_tot = 17  # 16 EMG + 1 Trigger/Spurious
        n_samples_per_chunk = 125
        bytes_per_chunk = 2 * num_ch_tot * n_samples_per_chunk
        conv_factor = 0.000286 * 1000  # Conversione in mV

        try:
            # Controllo connessione seriale
            if (
                self.ui is None 
                or not hasattr(self.ui, 'serialPort') 
                or self.ui.serialPort is None 
                or not hasattr(self.ui.serialPort, 'is_open')
                or not self.ui.serialPort.is_open
            ):
                raise RuntimeError("Hardware EAST non connesso! Connettere la porta COM prima di avviare.")

            # Resetta il buffer ed invia il comando di avvio streaming alla scheda STM32
            self.ui.serialPort.reset_input_buffer()
            self.ui.send_start_EMG_command()

            while self.is_running:
                if self.ui.serialPort.in_waiting >= bytes_per_chunk:
                    chunk_bytes = self.ui.serialPort.read(bytes_per_chunk)
                    if not chunk_bytes:
                        break

                    data_chunk = np.array(
                        struct.unpack(f"{len(chunk_bytes)//2}h", chunk_bytes),
                        dtype=np.float64
                    ).reshape((num_ch_tot, n_samples_per_chunk), order="F")

                    # Ri-allineamento del canale spurio
                    spurious_idx = int(np.argmin(np.var(data_chunk, axis=1)))
                    shift_back = (num_ch_tot - 1 - spurious_idx) % num_ch_tot
                    if shift_back != 0:
                        data_chunk = np.roll(data_chunk, shift_back, axis=0)

                    # Estrazione dei 16 canali EMG e conversione in mV
                    emg_16ch_chunk = data_chunk[:16, :] * conv_factor
                    self.signal_chunk.emit(emg_16ch_chunk)

                self.msleep(10)

        except Exception as e:
            self.signal_error.emit(str(e))
        finally:
            # Invio comando di stop al firmware STM32
            if self.ui and hasattr(self.ui, 'fcn_stop_stim') and self.ui.serialPort and self.ui.serialPort.is_open:
                try:
                    self.ui.fcn_stop_stim()
                except Exception:
                    pass