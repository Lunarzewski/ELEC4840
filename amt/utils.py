from soundfile import SoundFile
import librosa

SAMPLE_RATE = 44100
CHANNELS = 1
SUBTYPE = 'PCM_24'


def open_wav(path):
    wav_file = SoundFile(path)
    if wav_file.samplerate != SAMPLE_RATE:
        raise Exception("The sample rate of this wav file is incorrect, must be 44100kHz.")
    if wav_file.channels != CHANNELS:
        raise Exception("The number of channels in this wav file is incorrect, must be mono.")
    if wav_file.subtype != SUBTYPE:
        raise Exception("The bitrate of this wav file is incorrect, must be PCM-24 encoded.")

    return wav_file


def estimate_tempo(y, start_bpm=120):
    return librosa.beat.tempo(y=y, sr=SAMPLE_RATE, start_bpm=start_bpm)
