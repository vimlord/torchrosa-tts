
import csv

def add_word_to_phoneme(p, w, pho_to_words):
    if p in pho_to_words:
        pho_to_words[p].append(w)
    else:
        pho_to_words[p] = [w]

def load_phonemes(word, pho_to_words):
    from g2p_en import G2p
    g2p = G2p()

    for p in g2p(word):
        p = ''.join([c for c in p if not c.isdigit()])
        add_word_to_phoneme(p, word, pho_to_words)

def generate_metadata(in_fname='word_list.txt', out_fname='phoneme_samples.csv'):
    pho_to_words = {}

    # Read from file
    with open(in_fname) as f:
        reader = csv.reader(f)
        words = [r[0] for r in reader]
        print('Loaded word list')

    # Load words
    for w in words:
        load_phonemes(w, pho_to_words)

    # Sort
    for p in pho_to_words:
        pho_to_words[p] = sorted(pho_to_words[p])

    rows = []

    for p in sorted(list(pho_to_words.keys())):
        for w in pho_to_words[p]:
            rows.append({'phoneme': p, 'word' : w})

    # Write CSV
    with open(out_fname, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['phoneme', 'word'], delimiter=',')
        writer.writeheader()
        writer.writerows(rows)
        print('Saved phoneme to word mapping')


