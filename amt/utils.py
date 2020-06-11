from soundfile import SoundFile
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

from amt.plca import plca

SAMPLE_RATE = 44100
CHANNELS = 1
SUBTYPE = 'PCM_24'
HOP_LENGTH = 512
TIME_PER_FRAME = HOP_LENGTH / SAMPLE_RATE


def open_wav(path):
    wav_file = SoundFile(path)
    if wav_file.samplerate != SAMPLE_RATE:
        raise Exception("The sample rate of this wav file is incorrect, must be 44100kHz.")
    if wav_file.channels != CHANNELS:
        raise Exception("The number of channels in this wav file is incorrect, must be mono.")
    if wav_file.subtype != SUBTYPE:
        raise Exception("The bitrate of this wav file is incorrect, must be PCM-24 encoded.")

    return wav_file


def estimate_tempo(y, start_bpm=120.0):
    return librosa.beat.tempo(y=y, sr=SAMPLE_RATE, start_bpm=start_bpm)[0]


def estimate_piano_roll(y, tempo, plca_threshold, note_length_threshold):
    cqt = librosa.cqt(y,
                      sr=SAMPLE_RATE,
                      n_bins=60,
                      bins_per_octave=12,
                      fmin=librosa.note_to_hz('C2'))

    dictionary = np.load('dictionaries/piano_dictionary.npy')
    piano_roll = get_piano_roll(cqt, 60, dictionary, tempo, plca_threshold, note_length_threshold)
    return cqt, piano_roll


def get_piano_roll(cqt, number_of_notes, dictionary, tempo, plca_threshold, note_length_threshold):
    _, Pp_t = plca(cqt, number_of_notes, dictionary, maxstep=50)

    # Thresholding
    Pp_t[Pp_t < plca_threshold] = 0
    Pp_t[Pp_t >= plca_threshold] = 1

    # Get rid of frames lower than minimum
    min_frames = get_minimum_frames(tempo, note_length_threshold)
    Pp_t = threshold_minimum_frames(Pp_t, min_frames)
    return Pp_t


def threshold_minimum_frames(data_copy, min_frames):
    data = data_copy.copy()
    for i in range(np.shape(data)[0]):
        position_1 = 0
        position_2 = 0
        ones_length = 0
        for j in range(np.shape(data)[1]):
            if data[i, j] == 1:
                ones_length += 1
                if ones_length == 1:
                    position_1 = j
            if data[i, j] == 0 and ones_length != 0:
                position_2 = j
                if position_2 - position_1 < min_frames:
                    data[i, position_1:position_2] = 0
                position_1 = 0
                position_2 = 0
                ones_length = 0
        if ones_length != 0:
            position_2 = np.shape(data)[1]
            if position_2 - position_1 < min_frames:
                data[i, position_1:position_2] = 0
    return data


def get_minimum_frames(tempo, note_length_threshold):
    sixteenth_note_time = (60.0 / float(tempo)) / 4.0
    return round(sixteenth_note_time / TIME_PER_FRAME) - note_length_threshold


def estimate_onset_times(data, pre_max=6, post_max=6):
    return librosa.onset.onset_detect(y=data,
                                      sr=SAMPLE_RATE,
                                      units='frames',
                                      pre_max=pre_max,
                                      post_max=post_max)
                                      #pre_avg,
                                      #post_avg,
                                      #delta,
                                      #wait)


def smooth_onsets(data, onsets, onset_range=3, prev_note_range=8):
    for i in range(np.shape(data)[0]):
        one_mode = False
        for j in range(np.shape(data)[1]):
            if data[i, j] == 1 and not one_mode:
                one_mode = True
                closest_onset = onsets[(np.abs(onsets - j)).argmin()]
                if np.abs(np.min([closest_onset, j]) - np.max([closest_onset, j])) > onset_range:
                    if j != 0:
                        if j > prev_note_range:
                            if data[i, j - prev_note_range:j].max() == 1:
                                data[i, j - prev_note_range:j] = 1
                        else:
                            if data[i, 0:j].max() == 1:
                                data[i, j - prev_note_range:j] = 1
            if data[i, j] == 0 and one_mode:
                one_mode = False
