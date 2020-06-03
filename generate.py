
import sys

import vae

import argparse

parser = argparse.ArgumentParser(description="Train a text-to-speech model")

parser.add_argument("--method", default="vae",
        help="Model type to train")

parser.add_argument("--output", default="output.wav",
        help="The output location of the audio")

parser.add_argument("--text", required=True,
        help="The text to convert to speech")

parser.add_argument("--show-spectrogram", action='store_true',
        help="Shows the Mel spectrogram of the produced audio")

args = parser.parse_args()

method = args.method

if method == 'vae':
    vae.generate(args.text,
            out_fname=args.output,
            show_mel=args.show_spectrogram)
else:
    print(f'{sys.argv[0]}: Failed to generate speech via unknown method {method}')


