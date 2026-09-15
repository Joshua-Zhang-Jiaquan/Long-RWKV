def has_alternative_bit_fast(number: int) -> bool:
    """Check for alternating bits using O(1) bitmask arithmetic.

    Args:
        number: A positive integer to check.

    Returns:
        True if every pair of adjacent bits differs, False otherwise.

    Examples:
        >>> has_alternative_bit_fast(5)
        True
        >>> has_alternative_bit_fast(7)
        False
    """
    mask_even_bits = int("aaaaaaaa", 16)  # ...10101010
    mask_odd_bits = int("55555555", 16)  # ...01010101
    return mask_even_bits == (number + (number ^ mask_even_bits)) or mask_odd_bits == (
        number + (number ^ mask_odd_bits)
    )
