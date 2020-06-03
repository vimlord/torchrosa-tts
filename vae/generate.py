
import os

import torch

import numpy as np

import vae.train as train

from util.spectro import *
from util.preprocess import text_to_phonemes, phonemes_with_counts

from g2p_en import G2p

from vae.model import *

import matplotlib.pyplot as plt

from vae.hyperparameters import *

def scale(mel, lo=None, hi=None):
    if lo is None:
        lo = np.min(mel)
    if hi is None:
        hi = np.max(mel)
    
    # Rescale
    mel = (mel - np.min(mel)) / (np.max(mel) - np.min(mel))
    return lo + (hi - lo) * mel

def generate(text, out_fname = './output.wav', show_mel=False):
    
    # Load the model
    model_fname = './model.pt'
    if os.path.isfile(model_fname):
        mdl = torch.load(model_fname)
    else:
        mdl = train.run_pipeline()
        torch.save(mdl, model_fname)

    print('Loaded model from', model_fname)

    indices = text_to_phonemes(text, mdl.phonemes, WEIGHTS)
    indices, lengths = phonemes_with_counts(indices)
    
    # Each length is assigned to a specific scale
    lengths = [l * SCALE for l in lengths]

    idxs = torch.from_numpy(np.array(indices)).to(device)

    # Generate the spectrogram
    spectrogram = mdl.fabricate(idxs, lens=lengths).cpu().detach().numpy()
    spectrogram = scale(spectrogram, lo=-70, hi=35)
    
    if FORCE_BLANK_SILENCE:
        i = 0
        for x, j in enumerate(lengths):
            if indices[x] == 0:
                print(i, i+j)
                spectrogram[:,i:i+j] = 0
            i += j

    print('Spectrogram shape:', spectrogram.shape)

    if show_mel:
        show_spectrogram(spectrogram)
        plt.show()
    
    # Create a wave file
    wav = mel_to_wav(db_to_pow(spectrogram))

    print('WAV shape:', wav.shape)

    # Save the data
    save_audio_file(wav, out_fname)
    print('Saved file to', out_fname)


