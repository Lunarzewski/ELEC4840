from PyQt5 import QtCore, QtGui, QtWidgets, uic
import pyqtgraph as pg
import numpy as np
import librosa
from collections import OrderedDict

from amt.entities import Track
from amt.utils import open_wav

SAMPLE_RATE = 44100


class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('pyqtgraph.ui', self)
        self.setWindowTitle("Automatic Music Transcription")
        self.setGeometry(50, 50, 1116, 895)
        self.UiComponents()

        # Non UI Components
        self.track = Track()

    def UiComponents(self):
        # Import Wav File Button
        self.button_import_wav = QtWidgets.QPushButton("Import Wav File", self)
        self.button_import_wav.setGeometry(QtCore.QRect(200, 300, 100, 25))
        self.button_import_wav.clicked.connect(self.wav_file_open)

        # Import Wav File Textbox
        self.lineedit_import_wav = QtWidgets.QLineEdit(self)
        self.lineedit_import_wav.setGeometry(QtCore.QRect(300, 300, 200, 25))

        # Transcribe Button
        self.button_transcribe = QtWidgets.QPushButton("Transcribe", self)
        self.button_transcribe.setGeometry(QtCore.QRect(200, 400, 100, 25))
        self.button_transcribe.clicked.connect(self.transcribe)

        # Plot
        # Interpret image data as row-major instead of col-major
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.p1 = self.graphWidget.addPlot()
        self.img = pg.ImageItem()
        self.p1.addItem(self.img)

        # Cmap
        pos = np.array([0., 1., 0.5, 0.25, 0.75])
        color = np.array([[0, 255, 255, 255], [255, 255, 0, 255], [0, 0, 0, 255], (0, 0, 255, 255), (255, 0, 0, 255)],
                         dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        lut = cmap.getLookupTable(0.0, 1.0, 256)
        self.img.setLookupTable(lut)
        self.img.setLevels([-50, 40])

    def wav_file_open(self):
        file, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                        "Open Wav File",
                                                        "",
                                                        "Wav Files (*.wav)")
        if file:
            self.lineedit_import_wav.setText(file)

    def transcribe(self):
        try:
            wav_file_path = self.lineedit_import_wav.text()
            if wav_file_path[-4:] != '.wav':
                raise Exception('Imported file must be a .wav')
            else:
                wav_file = open_wav(path=wav_file_path)
                cqt, piano_roll = self.track.from_wav_file(wav_file)
                self.graph(np.abs(cqt))

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Wav Import Issue', str(e))

    def graph(self, data):
        self.img.setImage(data)
        fn, tn = np.shape(data)

        f = range(0, fn)
        t = range(0, tn)

        f_axis_dict = dict(enumerate(librosa.cqt_frequencies(60,
                                         fmin=librosa.note_to_hz('C2'),
                                         bins_per_octave=12)))
        t_values = librosa.frames_to_time(t, sr=SAMPLE_RATE)
        t_formatted = ['%.2f' % elem for elem in t_values]
        t_axis_dict = dict(enumerate(t_formatted))
        t_axis_dict = list(OrderedDict(sorted(t_axis_dict.items())).items())
        major_ticks = t_axis_dict[::tn//12]
        del t_axis_dict[::tn//12]
        minor_ticks = t_axis_dict

        newTicks = self.p1.getAxis('bottom')
        newTicks.setTicks([major_ticks, minor_ticks])

        self.img.scale(t[-1] / np.size(data, axis=1),
                       f[-1] / np.size(data, axis=0))
        # Limit panning/zooming to the spectrogram
        self.p1.setLimits(xMin=0, xMax=t[-1], yMin=0, yMax=f[-1])
        # Add labels to the axis
        self.p1.setLabel('bottom', "Time", units='s')
        # If you include the units, Pyqtgraph automatically scales the axis and adjusts the SI prefix (in this case kHz)
        self.p1.setLabel('left', "Frequency", units='Hz')
