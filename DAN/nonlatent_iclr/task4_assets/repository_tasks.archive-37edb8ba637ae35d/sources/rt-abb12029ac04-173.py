def hexagonal_num(n: int) -> int:
    """
    Returns nth hexagonal number
    >>> hexagonal_num(143)
    40755
    >>> hexagonal_num(21)
    861
    >>> hexagonal_num(10)
    190
    """
    return n * (2 * n - 1)
