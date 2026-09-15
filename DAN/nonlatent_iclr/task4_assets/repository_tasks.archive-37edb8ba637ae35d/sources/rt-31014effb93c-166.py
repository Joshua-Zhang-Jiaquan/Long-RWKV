def swap_pair(number: int) -> int:
    """Swap every pair of adjacent bits in an integer.

    Masks odd-positioned bits (0xAAAAAAAA), shifts them right by one,
    masks even-positioned bits (0x55555555), shifts them left by one,
    and merges the results.

    Args:
        number: A non-negative integer.

    Returns:
        The integer with each adjacent pair of bits swapped.

    Examples:
        >>> swap_pair(22)
        41
        >>> swap_pair(10)
        5
    """
    odd_bits = (number & int("AAAAAAAA", 16)) >> 1
    even_bits = (number & int("55555555", 16)) << 1
    return odd_bits | even_bits
