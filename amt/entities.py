from amt.utils import estimate_tempo


class Track:
    def __init__(self, tempo=None, key=None, notes=None):
        self.tempo = tempo
        self.key = key
        if notes is None:
            self.notes = []
        else:
            self.notes = notes

    def from_wav_file(self, wav_file):
        tempo = estimate_tempo(wav_file.read())
        print(tempo)
        #notes = analyse_notes
        #key = analyse_key