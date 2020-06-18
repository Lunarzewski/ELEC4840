from amt import utils
from amt.utils import estimate_tempo, estimate_piano_roll, estimate_onset_times, smooth_onsets, round_to_sixteenth, \
    rotate, Instrument
import numpy as np
import scipy.stats
import librosa
from enum import Enum
from midiutil import MIDIFile
import midiutil


def estimate_key(notes):
    pitch_class_dict = dict.fromkeys(utils.PITCH_CLASSES, 0)
    for note in notes:
        pitch_class_dict[note.pitch.get_pitch_class()] += note.duration.total_beat
    song_distribution = list(pitch_class_dict.values())
    maj_scores = []
    min_scores = []
    for i in range(12):
        maj_scores.append(scipy.stats.pearsonr(song_distribution, rotate(utils.KRUMHANSL_MAJ, i))[0])
        min_scores.append(scipy.stats.pearsonr(song_distribution, rotate(utils.KRUMHANSL_MIN, i))[0])
    max_score = np.argmax([maj_scores, min_scores])
    if max_score > 11:
        key = Key(pitch_class=KeyName(max_score-12), scale=Scale.MINOR)
    else:
        key = Key(pitch_class=KeyName(max_score), scale=Scale.MAJOR)
    return key


def notes_from_piano_roll(piano_roll, tempo):
    notes = []
    for i in range(np.shape(piano_roll)[0]):
        if np.sum(piano_roll[i, :]) != 0:
            one_mode = False
            start_frame = None
            for j in range(np.shape(piano_roll)[1]):
                if piano_roll[i, j] == 1 and not one_mode:
                    one_mode = True
                    start_frame = j
                if piano_roll[i, j] == 0 and one_mode:
                    one_mode = False
                    notes.append(
                        Note(Pitch(name=utils.NOTE_NAME_LIST[i]),
                             Duration(tempo=tempo, frames=(start_frame, j - start_frame))))
                    start_frame = None
            if one_mode:
                notes.append(
                    Note(Pitch(name=utils.NOTE_NAME_LIST[i]),
                         Duration(tempo=tempo, frames=(start_frame, (np.shape(piano_roll)[1] - 1) - start_frame))))
    return notes


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
        self.display_cqt = None
        self.instrument = None

    def from_wav_file(self, wav_file,
                      plca_threshold,
                      note_length_threshold,
                      onset_range,
                      previous_note_range,
                      pre_max,
                      post_max,
                      instrument):
        self.instrument = instrument
        self.samples = wav_file.read()
        self.tempo = int(round(estimate_tempo(self.samples)))
        self.cqt, self.piano_roll, self.display_cqt = estimate_piano_roll(self.samples, self.tempo, plca_threshold,
                                                                          note_length_threshold, instrument)
        self.onsets = estimate_onset_times(self.samples, pre_max=pre_max, post_max=post_max)
        smooth_onsets(self.piano_roll, self.onsets, onset_range=onset_range, prev_note_range=previous_note_range)
        if instrument == Instrument.GUITAR:
            self.piano_roll = np.pad(self.piano_roll, ((4, 8), (0, 0)))
        self.notes = notes_from_piano_roll(self.piano_roll, self.tempo)
        self.key = estimate_key(self.notes)

    def to_midi_file(self, file):
        midi = MIDIFile(1)
        midi.addTempo(0, 0, self.tempo)

        if self.key.scale == Scale.MAJOR:
            mode = midiutil.MAJOR
        else:
            mode = midiutil.MINOR
        if self.key.get_circle_of_fifths()[1] == 'Sharps':
            accidental_type = midiutil.SHARPS
        else:
            accidental_type = midiutil.FLATS
        midi.addKeySignature(0, 0, self.key.get_circle_of_fifths()[0], accidental_type, mode)

        for note in self.notes:
            midi.addNote(0, 0, note.pitch.midi_number, note.duration.start_beat * 4, note.duration.total_beat * 4,
                         utils.MIDI_VOLUME)
        with open(file, "wb") as output_file:
            midi.writeFile(output_file)


class Note:
    def __init__(self, pitch=None, duration=None):
        self.pitch = pitch
        self.duration = duration

    def __str__(self):
        return "{} at time {} for {}(s)".format(self.pitch.name, self.duration.start_time, self.duration.total_time)


class Pitch:
    def __init__(self, name=None, frequency=None, midi_number=None):
        if name is not None:
            self.name = name
        elif frequency is not None:
            self.frequency = frequency
        elif midi_number is not None:
            self.midi_number = midi_number

    def __str__(self):
        return "Name: {}, Frequency: {:.2f}, Midi Number: {}".format(self.name, self.frequency, self.midi_number)

    @property
    def name(self):
        return self.__name

    @property
    def frequency(self):
        return self.__frequency

    @property
    def midi_number(self):
        return self.__midi_number

    @name.setter
    def name(self, value):
        if value in utils.NOTE_NAME_LIST:
            self.__name = value
            index = utils.NOTE_NAME_LIST.index(value)
            self.__frequency = utils.NOTE_FREQ_LIST[index]
            self.__midi_number = utils.NOTE_MIDI_LIST[index]
        else:
            raise Exception("Note name {} does not exist".format(value))

    @frequency.setter
    def frequency(self, value):
        if value in utils.NOTE_FREQ_LIST:
            self.__frequency = value
            index = utils.NOTE_FREQ_LIST.index(value)
            self.__name = utils.NOTE_NAME_LIST[index]
            self.__midi_number = utils.NOTE_MIDI_LIST[index]
        else:
            raise Exception("Note frequency {} does not exist".format(value))

    @midi_number.setter
    def midi_number(self, value):
        if value in utils.NOTE_MIDI_LIST:
            self.__midi_number = value
            index = utils.NOTE_MIDI_LIST.index(value)
            self.__name = utils.NOTE_NAME_LIST[index]
            self.__frequency = utils.NOTE_FREQ_LIST[index]
        else:
            raise Exception("Note midi number {} does not exist".format(value))

    def get_pitch_class(self):
        return self.name[:-1]


class Duration:
    def __init__(self, tempo, time=None, beat=None, frames=None):
        self.tempo = tempo
        self.quarter_note_time = 60 / self.tempo
        self.beat_time_dict = {
            0.0625: self.quarter_note_time / 4,
            0.125: self.quarter_note_time / 2,
            0.25: self.quarter_note_time,
            0.5: self.quarter_note_time * 2,
            1.0: self.quarter_note_time * 4
        }
        if time is not None:
            self.start_time = time[0]
            self.total_time = time[1]
        elif beat is not None:
            self.start_beat = beat[0]
            self.total_beat = beat[1]
        elif frames is not None:
            self.start_frames = frames[0]
            self.total_frames = frames[1]

    def __str__(self):
        return "Begins at time {:.2f} for a total of {:.2f}(s)".format(self.start_time, self.total_time)

    @property
    def start_time(self):
        return self.__start_time

    @property
    def total_time(self):
        return self.__total_time

    @property
    def start_beat(self):
        return self.__start_beat

    @property
    def total_beat(self):
        return self.__total_beat

    @property
    def start_frames(self):
        return self.__start_frames

    @property
    def total_frames(self):
        return self.__total_frames

    @start_time.setter
    def start_time(self, value):
        self.__start_time = value
        # self.__start_beat = min(self.get_beat_time_dict().items(), key=lambda x: np.abs(x[1] - value))
        self.__start_beat = round_to_sixteenth(0.25 * (value / self.quarter_note_time))
        self.__start_frames = librosa.time_to_frames(value, sr=utils.SAMPLE_RATE)

    @total_time.setter
    def total_time(self, value):
        self.__total_time = value
        # self.__start_beat = min(self.get_beat_time_dict().items(), key=lambda x: np.abs(x[1] - value))
        self.__total_beat = round_to_sixteenth(0.25 * (value / self.quarter_note_time))
        self.__total_frames = librosa.time_to_frames(value, sr=utils.SAMPLE_RATE)

    @start_beat.setter
    def start_beat(self, value):
        self.__start_beat = value
        self.__start_time = value / self.quarter_note_time
        self.__start_frames = librosa.time_to_frames(self.__start_time, sr=utils.SAMPLE_RATE)

    @total_beat.setter
    def total_beat(self, value):
        self.__total_beat = value
        self.__total_time = value / self.quarter_note_time
        self.__total_frames = librosa.time_to_frames(self.__total_time, sr=utils.SAMPLE_RATE)

    @start_frames.setter
    def start_frames(self, value):
        self.__start_frames = value
        self.__start_time = librosa.frames_to_time(value, sr=utils.SAMPLE_RATE)
        self.__start_beat = round_to_sixteenth(0.25 * (self.__start_time / self.quarter_note_time))

    @total_frames.setter
    def total_frames(self, value):
        self.__total_frames = value
        self.__total_time = librosa.frames_to_time(value, sr=utils.SAMPLE_RATE)
        self.__total_beat = round_to_sixteenth(0.25 * (self.__total_time / self.quarter_note_time))


class Key:
    def __init__(self, pitch_class, scale):
        self.__pitch_class = pitch_class
        self.__scale = scale

    def __str__(self):
        return self.pitch_class.name.title() + " " + self.scale.name.title()

    @property
    def pitch_class(self):
        return self.__pitch_class

    @property
    def scale(self):
        return self.__scale

    @pitch_class.setter
    def pitch_class(self, value):
        self.__pitch_class = value
        self.name = self.pitch_class.name.title() + " " + self.scale.name.title()

    @scale.setter
    def scale(self, value):
        self.__scale = value
        self.name = self.pitch_class.name.title() + " " + self.scale.name.title()

    def to_display(self):
        return_str = str(self.pitch_class.name).split('_')
        if len(return_str) == 1:
            return return_str[0] + ' ' + self.scale.name.title()
        if return_str[1] == 'SHARP':
            return return_str[0] + '#' + ' ' + self.scale.name.title()
        else:
            return return_str[0] + '\u266d' + ' ' + self.scale.name.title()

    def get_circle_of_fifths(self):
        return utils.CIRCLE_OF_FIFTHS[str(self)]


class KeyName(Enum):
    C = 0
    C_SHARP = 1
    D = 2
    E_FLAT = 3
    E = 4
    F = 5
    F_SHARP = 6
    G = 7
    A_FLAT = 8
    A = 9
    B_FLAT = 10
    B = 11


class Scale(Enum):
    MAJOR = 0
    MINOR = 1
