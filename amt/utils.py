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


def estimate_notes(y, tempo):
    cqt = librosa.cqt(y,
                      sr=SAMPLE_RATE,
                      n_bins=60,
                      bins_per_octave=12,
                      fmin=librosa.note_to_hz('C2'))

    dictionary = np.load('dictionaries/piano_dictionary.npy')
    piano_roll = get_piano_roll(cqt, 60, dictionary, tempo)
    return cqt, piano_roll


def get_piano_roll(cqt, number_of_notes, dictionary, tempo):
    # TODO make threshold adjustable
    _, Pp_t = plca(cqt, number_of_notes, dictionary)

    # Thresholding
    Pp_t[Pp_t < 0.10] = 0
    Pp_t[Pp_t >= 0.10] = 1

    # Get rid of frames lower than minimum
    min_frames = get_minimum_frames(tempo)
    Pp_t = threshold_minimum_frames(Pp_t, min_frames)
    # librosa.display.specshow(Pp_t,
    #                          sr=44100,
    #                          fmin=librosa.note_to_hz('C2'),
    #                          x_axis='time',
    #                          y_axis='cqt_note')
    # # plt.vlines(onset_times, ymin=librosa.note_to_hz('C2'), ymax=librosa.note_to_hz('B6'), color='red', alpha=0.8)
    #
    # plt.show()
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


def get_minimum_frames(tempo):
    sixteenth_note_time = (60.0 / float(tempo)) / 4.0
    # TODO look into thresholding this?
    return round(sixteenth_note_time / TIME_PER_FRAME) - 2
