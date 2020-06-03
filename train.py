
import sys

import vae

import argparse

parser = argparse.ArgumentParser(description="Train a text-to-speech model")

parser.add_argument("--method", default="vae",
        help="Model type to train")

args = parser.parse_args()

method = args.method

if method == 'vae':
    vae.train()
else:
    print(f'{sys.argv[0]}: Failed to train model for method {method}')


