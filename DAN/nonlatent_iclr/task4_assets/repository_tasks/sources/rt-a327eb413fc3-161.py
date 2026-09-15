def count_flips_to_convert(first: int, second: int) -> int:
    """Count the number of bit flips needed to convert one integer to another.

    Args:
        first: The source integer.
        second: The target integer.

    Returns:
        The number of bits that differ between *first* and *second*.

    Examples:
        >>> count_flips_to_convert(29, 15)
        2
        >>> count_flips_to_convert(34, 34)
        0
    """
    diff = first ^ second
    count = 0
    while diff:
        diff &= diff - 1
        count += 1
    return count
