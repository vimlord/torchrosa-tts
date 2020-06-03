""" phoneme_handler.py

Provides utilities for processing words and phonemes for annotation.
"""

import csv
import os
import sys
import time
import uuid

import util.csv_helper as csv_helper

_g2p = None
def g2p(x):
    from g2p_en import G2p

    global _g2p
    if _g2p is None:
        _g2p = G2p()
    return _g2p(x)
    

def list_words(*phonemes):
    from g2p_en import G2p

    with open('phoneme_samples.csv') as f:
        reader = csv.DictReader(f, delimiter=',')

        rows = [row for row in reader]

    if len(phonemes) == 0:
        for row in rows:
            print(row['word'])
        return
    
    for phoneme in phonemes:
        if len(phonemes) > 1:
            print(phoneme)

        g2p = G2p()

        for row in rows:
            pho = row['phoneme']
            word = row['word']

            phos = g2p(word)

            doc = f"{word} [{' '.join(phos)}]"

            if pho == phoneme:
                if len(phonemes) == 1:
                    print(doc)
                else:
                    print(f'\t{doc}')

def list_phonemes(*args):
    with open('phoneme_samples.csv') as f:
        reader = csv.DictReader(f, delimiter=',')

        phos = set([])

        for row in reader:
            phos.add(row['phoneme'])

        for pho in sorted(list(phos)):
            print(pho)

def phonemize(*words):
    for w in words:
        print(w, '->', g2p(w))

def generate_metadata(*args):
    if len(args) == 0:
        csv_helper.generate_metadata()
    elif len(args) == 2:
        csv_helper.generate_metadata(*args)
    else:
        print(f'{sys.argv[0]} {sys.argv[1]}: either no args or two args should be supplied')

def record_phoneme(phoneme, path='./phoneme-data/samples', length=5):
    import pyaudio
    import wave
    
    print(f'Examples of phoneme {phoneme}:')
    list_words(phoneme)

    h = uuid.uuid4()
    path = os.path.join(path, phoneme)
    fname = os.path.join(path, f'{h}.wav')
    
    # Wait for confirmation, then a second after
    input('Press enter to start recording in 1 second')
    print('Waiting 1s...')
    time.sleep(1)
 
    p = pyaudio.PyAudio()
    stream = p.open(
            format=pyaudio.paInt16,
            channels=2,
            rate=44100,
            input=True,
            frames_per_buffer=1024)
    
    # Do the recording
    print('Start')
    frames = [stream.read(1024) for _ in range(int(44100 * length / 1024))]
    print('End')
    
    # Close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Write the file
    if not os.path.exists(path):
        print('Creating directory', path)
        os.makedirs(path)

    print('Writing sample of', phoneme, 'to', fname)

    wf = wave.open(fname, 'wb')
    wf.setnchannels(2)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(44100)
    wf.writeframes(b''.join(frames))

options = {
    'list-words' : list_words,
    'list-phonemes' : list_phonemes,
    'generate-metadata' : generate_metadata,
    'phonemize' : phonemize,
    'record-phoneme' : record_phoneme,
}

help_options = {
    'list-words' : "Lists the words matching the given phonemes, or all if no phonemes given",
    'list-phonemes' : "Lists the phonemes available in the CSV metadata",
    'generate-metadata' : "Generates the CSV metadata, optionally taking both the word list and output location",
    'phonemize' : "Given words, uses g2p to generate the phonemes",
    'record-phoneme' : "Given a phoneme, uses the microphone to record a five second segment of audio for that phoneme",
    'help' : "Displays help options"
}

def print_options():
    print('Options:')
    for c in sorted(options):
        print(f' {c}')
        print(f'        {help_options[c]}')

options['help'] = print_options

if len(sys.argv) < 2:
    print(f'{sys.argv[0]}: no command supplied')
    print_options()
elif sys.argv[1] not in options:
    print(f'{sys.argv[0]}: command {cmd} not recognized')
    print_options()
else:
    options[sys.argv[1]](*sys.argv[2:])

