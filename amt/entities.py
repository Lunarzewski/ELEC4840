from amt.utils import estimate_tempo, estimate_notes
import numpy as np


class Track:
    def __init__(self, tempo=None, key=None, notes=None):
        self.tempo = tempo
        self.key = key
        if notes is None:
            self.notes = []
        else:
            self.notes = notes

    def from_wav_file(self, wav_file):
        samples = wav_file.read()
        self.tempo = int(round(estimate_tempo(samples)))
        estimate_notes(samples, self.tempo)
        #notes = analyse_notes
        #key = analyse_key