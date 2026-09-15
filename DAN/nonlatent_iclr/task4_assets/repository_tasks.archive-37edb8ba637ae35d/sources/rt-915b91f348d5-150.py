def insert_one_bit(number: int, bit: int, position: int) -> int:
    """Insert a single bit at a specific position in an integer.

    Splits the number at *position*, shifts the upper part left by one
    to make room, inserts *bit*, and merges with the lower part.

    Args:
        number: The integer to modify.
        bit: The bit value to insert (0 or 1).
        position: Zero-based index at which to insert the bit.

    Returns:
        The resulting integer after insertion.

    Examples:
        >>> insert_one_bit(21, 1, 2)
        45
        >>> insert_one_bit(21, 0, 2)
        41
    """
    upper = number >> position
    upper = (upper << 1) | bit
    upper = upper << position
    lower = ((1 << position) - 1) & number
    return lower | upper
