
import librosa
import librosa.display

import os
import sys

import numpy as np
import scipy

import matplotlib.pyplot as plt

from util.spectro import *

def file_to_spectrogram(fname, builder=wav_to_fft, to_db=amp_to_db, **args):
    wav = load_audio_file(fname)
    spc = builder(wav, **args)
    return to_db(spc)

def preview_spectrogram(fname, hop_length=512):
    spc = file_to_spectrogram(fname, hop_length=hop_length)
    print(spc.shape)
    show_spectrogram(spc, hop_length=hop_length)

def preview(fname, hop_length=512):
    plt.subplot(1, 1, 1)
    preview_spectrogram(fname, hop_length=hop_length)
    plt.show()
 
#hop_length = 2 ** int(math.log(22050 / 20)) # 1024
HOP_LENGTH = 512

USE_MEL=True

def pipeline(
        in_dir='./phone,e_data/samples',
        out_dir='./phoneme-data/spectrograms'):
    """ Loads WAV files and converts them to spectrograms.
    
    The file structure of the WAV files should be as follows:

    in_dir/
        AA/
            file1.wav
            file2.wav
            ...
        AE/
            file1.wav
            file2.wav
            ...
        ...

    This will generate an equivalent directory structure like so:
    
    out_dir/
        AA/
            file1.npy
            file2.npy
            ...
        AE/
            file1.npy
            file2.npy
            ...
        ...

    in_dir
            The root location of the WAV files on disk, organized by phoneme.
    out_dir
            The root directory to save the spectrograms to.
    """

    ct = 0
    for p in os.listdir(in_dir):
        path = os.path.join(in_dir, p)
        print('Searching', path, 'for recordings of phoneme', p)
        
        # Ensure output directory exists before continuing
        out_path = os.path.join(out_dir, p)
        os.makedirs(out_path, exist_ok=True)

        for f in os.listdir(path):
            if f[-4:] != '.wav': continue
            ct += 1
            
            fname = os.path.join(path, f)
            print('Converting', fname, 'to spectrogram')
            
            if USE_MEL:
                spc = file_to_spectrogram(fname,
                        builder=wav_to_mel,
                        to_db=pow_to_db,
                        hop_length=HOP_LENGTH)
            else:
                spc = file_to_spectrogram(fname, 
                        builder=wav_to_fft,
                        to_db=amp_to_db,
                        hop_length=HOP_LENGTH)
            
            
            h = f[:-4]
            fname = os.path.join(out_path, f'{h}.npy')
            np.save(fname, spc)
            print('Saved result to', fname)

    print('Files converted:', ct)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        in_dir = './phoneme-data/samples'
        out_dir = './phoneme-data/spectrograms'
    else:
        in_dir = sys.argv[1]
        out_dir = sys.argv[2]   

    pipeline(in_dir=in_dir, out_dir=out_dir)

