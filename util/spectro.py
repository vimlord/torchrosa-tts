""" spectro.py

Provides wrappers for working with audio files and spectrograms.
"""

import librosa
import librosa.display

import matplotlib.pyplot as plt

def load_audio_file(fname):
    y, _ = librosa.load(fname, sr=22050)
    return y

def save_audio_file(y, fname, sr=22050):
    librosa.output.write_wav(fname, y, sr=sr)

def amp_to_db(x, **args):
    return librosa.amplitude_to_db(x, **args)

def db_to_amp(x, **args):
    return librosa.db_to_amplitude(x, **args)

def pow_to_db(x, **args):
    return librosa.power_to_db(x, **args)

def db_to_pow(x, **args):
    return librosa.db_to_power(x, **args)

def wav_to_fft(wav, n_fft=2048, hop_length=512):
    return librosa.stft(wav,
            n_fft=n_fft,
            hop_length=hop_length)

def fft_to_wav(fft, hop_length=512):
    return librosa.istft(fft, hop_length=hop_length)

def wav_to_mel(wav, n_fft=2048, n_mels=128, hop_length=512):
    return librosa.feature.melspectrogram(y=wav, sr=22050,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length)

def mel_to_wav(mel, hop_length=512):
    return librosa.feature.inverse.mel_to_audio(mel)

def show_spectrogram(spc, hop_length=512):
    librosa.display.specshow(spc,
        sr=22050, hop_length=hop_length, x_axis='time', y_axis='log',
        cmap='hot')
    plt.colorbar(format='%+2.0f dB', cmap='hot')


