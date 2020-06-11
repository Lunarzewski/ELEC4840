from PyQt5 import QtCore, QtGui, QtWidgets, uic
import pyqtgraph as pg
import numpy as np
import librosa

from amt import utils
from amt.entities import Track
from amt.utils import open_wav


class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('pyqtgraph.ui', self)
        self.setWindowTitle("Automatic Music Transcription")
        self.setGeometry(50, 50, 1116, 895)
        self.UiComponents()
        self.set_defaults()

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
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img)
        self.graphWidget.addItem(self.hist)
        self.hist.gradient.restoreState(
            {'mode': 'rgb',
             'ticks': [(1.0, (254, 0, 2, 255)),
                       (0.75, (255, 240, 1, 255)),
                       (0.25, (152, 255, 77, 255)),
                       (0.0, (3, 2, 252, 255))]})

        # Output view Groupbox
        # PLCA Radio Button
        self.radio_plca_option = QtWidgets.QRadioButton("Piano Roll View", self)
        self.radio_plca_option.toggled.connect(self.toggle_output_view)

        # CQT Radio Button
        self.radio_cqt_option = QtWidgets.QRadioButton("CQT View", self)
        self.radio_cqt_option.setChecked(True)
        self.radio_cqt_option.toggled.connect(self.toggle_output_view)

        # View Onset Lines Checkbox
        self.checkbox_view_onset_lines = QtWidgets.QCheckBox("View Onsets", self)
        self.checkbox_view_onset_lines.toggled.connect(self.toggle_onset_lines)

        self.groupbox_output_view = QtWidgets.QGroupBox("Output View Options", self)
        vbox_output_view = QtWidgets.QVBoxLayout()
        vbox_output_view.addWidget(self.radio_cqt_option)
        vbox_output_view.addWidget(self.radio_plca_option)
        vbox_output_view.addWidget(self.checkbox_view_onset_lines)
        self.groupbox_output_view.setLayout(vbox_output_view)
        self.groupbox_output_view.setGeometry(QtCore.QRect(920, 110, 150, 150))

        # Y Axis Parameter Group Box
        # PLCA Radio Button
        self.radio_y_frequency = QtWidgets.QRadioButton("Frequency", self)
        self.radio_y_frequency.setChecked(True)
        self.radio_y_frequency.toggled.connect(self.toggle_y_axis)

        # CQT Radio Button
        self.radio_y_note = QtWidgets.QRadioButton("Note", self)
        self.radio_y_note.toggled.connect(self.toggle_y_axis)

        self.groupbox_y_axis_parameter = QtWidgets.QGroupBox("Y-Axis", self)
        vbox_y_axis_parameter = QtWidgets.QVBoxLayout()
        vbox_y_axis_parameter.addWidget(self.radio_y_frequency)
        vbox_y_axis_parameter.addWidget(self.radio_y_note)
        self.groupbox_y_axis_parameter.setLayout(vbox_y_axis_parameter)
        self.groupbox_y_axis_parameter.setGeometry(QtCore.QRect(920, 10, 150, 100))

    def set_defaults(self):
        self.p1.setLimits(xMin=0, xMax=50, yMin=0, yMax=59)
        self.p1.setLabel('left', "Frequency", units='Hz')
        self.p1.setLabel('bottom', "Time", units='s')
        self.toggle_y_axis()
        self.vline_objects = []

    def toggle_onset_lines(self):
        if not self.vline_objects:
            return

        if self.checkbox_view_onset_lines.isChecked():
            for line in self.vline_objects:
                line.show()
        if not self.checkbox_view_onset_lines.isChecked():
            for line in self.vline_objects:
                line.hide()

    def toggle_y_axis(self):
        freqs = librosa.cqt_frequencies(60,
                                        fmin=librosa.note_to_hz('C2'),
                                        bins_per_octave=12)
        f_axis_dict = []
        if self.radio_y_frequency.isChecked():
            freqs_formatted = ['%.2f' % elem for elem in freqs]
            f_axis_dict = list(dict(enumerate(freqs_formatted)).items())
            self.p1.setLabel('left', "Frequency", units='Hz')
        elif self.radio_y_note.isChecked():
            notes = librosa.hz_to_note(freqs)
            f_axis_dict = list(dict(enumerate(notes)).items())
            self.p1.setLabel('left', "Notes", units='')

        major_f_ticks = f_axis_dict[::60 // 4]
        del f_axis_dict[::60 // 4]
        minor_f_ticks = f_axis_dict
        newLeftTicks = self.p1.getAxis('left')
        newLeftTicks.setTicks([major_f_ticks, minor_f_ticks])

    def toggle_output_view(self):
        if self.radio_cqt_option.isChecked() and self.track.cqt is not None:
            self.graph(np.abs(self.track.cqt))
        elif self.radio_plca_option.isChecked() and self.track.piano_roll is not None:
            self.graph(self.track.piano_roll)

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
                self.track.from_wav_file(wav_file)
                self.toggle_output_view()
                # Graph onset lines
                self.graph_onset_lines()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Wav Import Issue', str(e))

    def graph(self, data):
        self.img.setImage(data)
        fn, tn = np.shape(data)

        f = range(0, fn)
        t = range(0, tn)

        t_values = librosa.frames_to_time(t, sr=utils.SAMPLE_RATE)
        t_formatted = ['%.2f' % elem for elem in t_values]
        t_axis_dict = list(dict(enumerate(t_formatted)).items())
        major_t_ticks = t_axis_dict[::tn // 12]
        del t_axis_dict[::tn // 12]
        minor_t_ticks = t_axis_dict
        newBottomTicks = self.p1.getAxis('bottom')
        newBottomTicks.setTicks([major_t_ticks, minor_t_ticks])

        # Limit panning/zooming to the spectrogram
        self.p1.setLimits(xMin=0, xMax=t[-1], yMin=0, yMax=f[-1])

    def graph_onset_lines(self):
        # Remove current lines
        if self.vline_objects:
            for line in self.vline_objects:
                self.p1.removeItem(line)
            self.vline_objects = []

        # No onsets, do nothing
        if self.track.onsets.size == 0:
            return

        for onset in self.track.onsets:
            vline = pg.InfiniteLine(onset)
            self.vline_objects.append(vline)
            if not self.checkbox_view_onset_lines.isChecked():
                vline.hide()
            self.p1.addItem(vline)
