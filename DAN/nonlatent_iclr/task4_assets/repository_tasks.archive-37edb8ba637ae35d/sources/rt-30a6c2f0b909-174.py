def is_pentagonal(n: int) -> bool:
    """
    Returns True if n is pentagonal, False otherwise.
    >>> is_pentagonal(330)
    True
    >>> is_pentagonal(7683)
    False
    >>> is_pentagonal(2380)
    True
    """
    root = (1 + 24 * n) ** 0.5
    return ((1 + root) / 6) % 1 == 0
