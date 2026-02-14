#!/usr/bin/env python3

import math

from utils import range_0_to_inclusive


def comb(n, k):
    assert k >= 0
    if n < k:
        return 0
    return math.comb(n, k)


# ── Implementation ────────────────────────────────────────────
def combination_count(identical_balls, distinct_urns, urn_volume=None):
    urn_volume = identical_balls if urn_volume is None else urn_volume

    # Special case: too many balls
    if identical_balls > distinct_urns * urn_volume:
        return 0

    def count_k_overloaded_urns(k):
        sign = (-1) ** k
        assert abs(sign) == 1
        urn_combination = comb(distinct_urns, k)
        overloaded_balls = k * (urn_volume + 1)
        assert overloaded_balls <= identical_balls
        balls_leftover = identical_balls - overloaded_balls
        assert balls_leftover >= 0
        overload_combinations = comb(
            balls_leftover + distinct_urns - 1, distinct_urns - 1
        )
        if overloaded_balls == identical_balls:
            assert overload_combinations == 1
        result = sign * urn_combination * overload_combinations
        return result

    return sum(
        map(
            count_k_overloaded_urns,
            range_0_to_inclusive(identical_balls // (urn_volume + 1)),
        )
    )


def test():
    for (b, u, m), expected in [
        [[3, 2, 1], 0],
        [[2, 2, 2], 3],
        [[3, 3, 2], 7],
        [[5, 4, 2], 16],
    ]:
        actual = combination_count(b, u, m)
        print(f"b={b} u={u} m={m} | expected={expected} | actual={actual}")
        assert actual == expected
    print("Tests OK")


if __name__ == "__main__":
    test()

    talents = int(input("how many talents? "))
    stars = int(input("how many stars? "))
    maximum = int(input("what is the star maximum? "))
    print(
        f"Distinct assignments: {combination_count(identical_balls=stars, distinct_urns=talents, urn_volume=maximum)}"
    )
