"""Small per-item sentence graph. Sentences that name the item are linked to the sentences that follow them in the same section."""
import numpy as np


def expand_with_followers(candidates, mention, sections, window):
    """All arrays cover one note in reading order. Returns a mask of candidates plus followers of candidate mentions."""
    out = candidates.copy()
    for i in np.flatnonzero(candidates & mention):
        for j in range(i + 1, min(i + window + 1, len(out))):
            if sections[j] != sections[i]:
                break
            out[j] = True
    return out
