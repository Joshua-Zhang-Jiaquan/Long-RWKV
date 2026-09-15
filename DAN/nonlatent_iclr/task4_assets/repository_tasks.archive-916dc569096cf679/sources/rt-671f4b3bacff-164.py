def count_ones_iter(number: int) -> int:
    """Count set bits using Brian Kernighan's algorithm (iterative).

    Args:
        number: A non-negative integer.

    Returns:
        The number of 1-bits in the binary representation.

    Examples:
        >>> count_ones_iter(8)
        1
        >>> count_ones_iter(63)
        6
    """
    count = 0
    while number:
        number &= number - 1
        count += 1
    return count
