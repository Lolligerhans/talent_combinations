#!/usr/bin/env python3

import functools
import random
from collections import Counter
from typing import List

from talents import combination_count
from talents import range_0_to_inclusive as range_0_to_inclusive


def precompute_dp(b: int, u: int, m: int) -> List[List[int]]:
    """
    Precompute DP table where dp[balls][urns] = number of ways
    to place 'balls' balls into 'urns' urns with capacity m.

    This uses O(b * u) memory and O(b * u) time.
    """
    assert b >= 0
    assert u >= 0
    assert m >= 0

    @functools.cache
    def dp(b, u, m):
        assert u >= 0
        assert b >= 0
        result = None
        if b == 0:
            result = 1
        else:
            if u == 0:
                result = 0
            else:
                result = sum(
                    [dp(b - k, u - 1, m) for k in range_0_to_inclusive(min(m, b))]
                )
        # print(f"dp({b},{u},{m}) == {result}")
        return result

    result = [
        [dp(balls, urns, m) for urns in range_0_to_inclusive(u)]
        for balls in range_0_to_inclusive(b)
    ]
    print(f"filled result = {result}")
    return result


def sample_configuration_dp(b: int, u: int, m: int, dp: List[List[int]]) -> List[int]:
    """
    Sample a configuration uniformly using the precomputed DP table.

    This runs in O(u * min(m, b)) time.
    """
    config = [-1] * u
    remaining_balls = b
    remaining_urns = u

    for urn_index in range(u):
        # For the current urn, choose how many balls it gets
        max_balls = min(m, remaining_balls)

        # Compute weights for each possible count
        weights = []
        for balls_in_urn in range(max_balls + 1):
            # Number of ways to complete with remaining_balls - balls_in_urn balls
            # into remaining_urns - 1 urns
            ways = dp[remaining_balls - balls_in_urn][remaining_urns - 1]
            weights.append(ways)

        # Choose based on weights
        total = sum(weights)
        r = random.random() * total
        cum = 0
        chosen = max_balls  # default fallback

        for i, w in enumerate(weights):
            cum += w
            if r < cum:
                chosen = i
                break

        config[urn_index] = chosen
        remaining_balls -= chosen
        remaining_urns -= 1

    return config


def sample_configuration_efficient(b: int, u: int, m: int) -> List[int]:
    """
    One-call function: sample a uniform configuration.
    """
    dp = precompute_dp(b, u, m)
    return sample_configuration_dp(b, u, m, dp)


def generate_placement_sequence(config: List[int]) -> List[int]:
    """
    Given a configuration, generate a random placement sequence.

    Creates a list where urn i appears config[i] times, then shuffles.
    """
    sequence = []
    for urn, count in enumerate(config):
        sequence.extend([urn] * count)
    random.shuffle(sequence)
    return sequence


def test_uniformity(b: int, u: int, m: int, samples: int = 100000) -> None:
    """Test that the sampling is uniform."""
    from collections import Counter

    print(f"Testing uniformity for b={b}, u={u}, m={m}, samples={samples}")
    print(f"Number of possible configurations: {combination_count(b, u, m)}")

    dp = precompute_dp(b, u, m)

    # Generate samples
    sampled_configs = []
    for _ in range(samples):
        config = sample_configuration_dp(b, u, m, dp)
        sampled_configs.append(tuple(config))

    # Count frequencies
    freq = Counter(sampled_configs)

    # Expected probability
    total_configs = combination_count(b, u, m)
    expected_prob = 1 / total_configs

    # Check a few configurations
    print(f"\nSampled {len(freq)} unique configurations out of {total_configs}")
    print("First few configurations and their empirical probabilities:")

    for config, count in list(freq.items())[:15]:
        empirical = count / samples
        print(f"  {config}: {empirical:.6f} (expected: {expected_prob:.6f})")

    # Chi-square test
    chi2 = 0
    for count in freq.values():
        expected = samples * expected_prob
        chi2 += (count - expected) ** 2 / expected

    print(f"\nChi-square statistic: {chi2:.2f}")
    print(f"Degrees of freedom: {len(freq) - 1}")
    print("(Lower values indicate better uniformity)")


def compare_with_bruteforce(b: int, u: int, m: int) -> None:
    """Compare DP sampling with brute-force enumeration for small cases."""
    from itertools import product

    # Generate all configurations
    all_configs = []
    for config in product(range(m + 1), repeat=u):
        if sum(config) == b:
            all_configs.append(config)

    print(f"Brute-force found {len(all_configs)} configurations")
    print(f"Inclusion-exclusion gives {combination_count(b, u, m)}")

    # Test DP sampling
    dp = precompute_dp(b, u, m)

    # Sample many times and compare frequencies
    samples = min(100000, 100 * len(all_configs))
    freq = Counter()
    for _ in range(samples):
        config = tuple(sample_configuration_dp(b, u, m, dp))
        freq[config] += 1

    # Check if all configurations appear
    missing = set(all_configs) - set(freq.keys())
    if missing:
        print(f"Warning: {len(missing)} configurations never sampled")

    # Check uniformity
    expected = samples / len(all_configs)
    max_deviation = max(abs(count - expected) / expected for count in freq.values())
    print(f"Maximum relative deviation from uniform: {max_deviation:.4f}")


def print_example(b, u, m):
    print(f"=== Example: b={b}, u={u}, m={m} ===")
    compare_with_bruteforce(b, u, m)

    # Compute total number of configurations
    total = combination_count(b, u, m)
    print(f"Total configurations: {total}")

    # Sample a configuration
    config = sample_configuration_efficient(b, u, m)
    print(f"Sampled configuration: {config}")

    # Generate a placement sequence
    sequence = generate_placement_sequence(config)
    print(f"Random placement sequence: {sequence[:10]}...")

    # Test uniformity
    print("\n--- Testing uniformity ---")
    test_uniformity(b, u, m, samples=1000000)


if __name__ == "__main__":
    # Example 1: Small test case
    print_example(3, 3, 2)
    print_example(5, 4, 2)
    print_example(10, 5, 4)
