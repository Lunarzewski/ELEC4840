from PyQt5 import QtCore, QtGui, QtWidgets

from amt.entities import Track
from amt.utils import open_wav


class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Automatic Music Transcription")
        self.setGeometry(50, 50, 1116, 895)

        self.UiComponents()

        # Non UI Components
        self.track = Track()

    def UiComponents(self):
        # Import Wav File Button
        self.button_import_wav = QtWidgets.QPushButton("Import Wav File", self)
        self.button_import_wav.setGeometry(QtCore.QRect(50, 50, 100, 25))
        self.button_import_wav.clicked.connect(self.wav_file_open)

        # Import Wav File Textbox
        self.lineedit_import_wav = QtWidgets.QLineEdit(self)
        self.lineedit_import_wav.setGeometry(QtCore.QRect(150, 50, 200, 25))

        # Transcribe Button
        self.button_transcribe = QtWidgets.QPushButton("Transcribe", self)
        self.button_transcribe.setGeometry(QtCore.QRect(200, 200, 100, 25))
        self.button_transcribe.clicked.connect(self.transcribe)

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
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Wav Import Issue', str(e))
