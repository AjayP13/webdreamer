import numpy as np

# DTW implementation
from fastdtw import dtw
from scipy.spatial.distance import cosine

from .symbolicai import embed_sequence, to_symbol


def compute_vertex_dtw(
    reference_sequence, test_sequence, baseline_sequence, discount=True
):
    score = 0
    # align sequences using DTW
    _, path = dtw(
        embed_sequence(reference_sequence), embed_sequence(test_sequence), dist=cosine
    )

    # iterate over the path and compute the score
    for idx in path:
        # similarity of embeddings
        s = min(
            max(reference_sequence[idx[0]].measure(test_sequence[idx[1]]).value, 0), 1
        )

        # baseline correction:
        if len(baseline_sequence) > idx[0]:
            rnd_score = (
                reference_sequence[idx[0]].measure(baseline_sequence[idx[0]]).value
            )
        else:
            rnd_score = 0
        s = s - rnd_score

        # distance discount
        if discount:
            s = 1 / (1 + np.abs(idx[0] - idx[1])) * s

        score += s
    return 1 / len(path) * score


def vertex_dtw(references, baseline, result):
    scores = []
    baseline_sequence = to_symbol(baseline["prompts"])
    test_sequence = to_symbol(result["prompts"])
    # compute DTW score for each reference
    for i in range(len(references)):
        reference_sequence = to_symbol(references[i]["prompts"])
        scores.append(
            compute_vertex_dtw(
                reference_sequence, test_sequence, baseline_sequence, discount=True
            )
        )
    # return highest score
    return max(scores)
