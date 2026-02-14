import random


def weighted_sample_index(weights):
    """Sample according to weights. Return index of sampled element."""
    total = sum(weights)
    r = random.random() * total

    cum = 0
    for i, w in enumerate(weights):
        cum += w
        if r < cum:
            return i

    # Default to hedge for float problems just below 1.0
    return len(weights) - 1
