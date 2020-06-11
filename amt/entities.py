from amt.utils import estimate_tempo, estimate_piano_roll, estimate_onset_times, smooth_onsets
import numpy as np


class Track:
    def __init__(self, tempo=None, key=None, notes=None):
        self.tempo = tempo
        self.key = key
        if notes is None:
            self.notes = []
        else:
            self.notes = notes
        self.cqt = None
        self.piano_roll = None
        self.onsets = []
        self.samples = None

    def from_wav_file(self, wav_file,
                      plca_threshold,
                      note_length_threshold,
                      onset_range,
                      previous_note_range,
                      pre_max,
                      post_max):
        self.samples = wav_file.read()
        self.tempo = int(round(estimate_tempo(self.samples)))
        self.cqt, self.piano_roll = estimate_piano_roll(self.samples, self.tempo, plca_threshold, note_length_threshold)
        self.onsets = estimate_onset_times(self.samples, pre_max=pre_max, post_max=post_max)
        smooth_onsets(self.piano_roll, self.onsets, onset_range=onset_range, prev_note_range=previous_note_range)
        #notes = analyse_notes
        #key = analyse_key
