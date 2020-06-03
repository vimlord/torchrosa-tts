
from g2p_en import G2p

def text_to_phonemes(text, phonemes, weights):
    """ Given text, converts to phonemes such that each
    one should be played for an equal amount of time.
    """
    # Get phonemes
    g2p = G2p()
    string = [' '] + g2p(text) + [' ']
    print('Extracted raw phonemes')

    indices = []
    for ph in string:
        p = ''.join(c for c in ph if not c.isdigit())

        w = -1 if not ph[-1].isdigit() else int(ph[-1])
        weight = weights[w]

        if p == 'AW':
            indices += ['AH'] * weight + ['W'] * weight
        elif p == 'AY':
            indices += ['AA'] * weight + ['IY'] * weight
        elif p == 'EY':
            indices += ['EH'] * weight + ['IY'] * weight
        elif p == 'NG':
            indices += ['N'] * weight + ['G'] * weight
        elif p == 'OW':
            indices += ['AH'] * weight + ['W'] * weight
        elif p == 'OY':
            indices += ['O'] * weight + ['IY'] * weight
        elif p == ' ':
            indices += ['_'] * 3
        elif '.' in p or p in '?!;,':
            pass
        else:
            indices += [p] * weight

    print('Trimmed phonemes')

    # Convert to indices
    return [phonemes.index(p) for p in indices]

def phonemes_with_counts(phonemes):
    """ Given loose phonemes, converts to a list of
    phonemes and how many occurred in a row. This is
    used to ensure generation of smoother phonemes.
    """

    ps, cts = [phonemes[0]], [1]

    for p in phonemes[1:]:
        if p == ps[-1]:
            cts[-1] += 1
        else:
            ps.append(p)
            cts.append(1)

    return ps, cts

