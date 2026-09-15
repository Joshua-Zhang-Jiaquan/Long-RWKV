def increment_score(count: int) -> int:
    """
    Calculates the score for a move based on the number of elements removed.

    >>> increment_score(3)
    6
    >>> increment_score(0)
    0
    """
    return int(count * (count + 1) / 2)
