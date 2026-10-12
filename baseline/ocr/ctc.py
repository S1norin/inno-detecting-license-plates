"""Greedy CTC decoding with a probability for each emitted character."""

import math


def decode_greedy(indices, probabilities, characters, blank=0):
    if len(indices) != len(probabilities):
        raise ValueError("CTC indices and probabilities have different lengths")

    text = []
    confidences = []
    previous = blank
    for index, probability in zip(indices, probabilities, strict=True):
        if not 0 <= index < len(characters):
            raise ValueError("CTC index outside vocabulary")
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid CTC probability")

        if index != blank:
            if index != previous:
                text.append(characters[index])
                confidences.append(float(probability))
            else:
                confidences[-1] = max(confidences[-1], float(probability))
        previous = index

    return {"text": "".join(text), "char_confidences": confidences}
