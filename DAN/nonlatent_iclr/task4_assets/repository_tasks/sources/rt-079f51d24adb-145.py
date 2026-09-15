def update_bit(number: int, position: int, bit: int) -> int:
    """Update the bit at a specific position to a given value.

    First clears the bit at *position*, then ORs in the new *bit* value
    shifted to that position.

    Args:
        number: The integer to modify.
        position: Zero-based bit index to update.
        bit: The new bit value (0 or 1).

    Returns:
        The integer with the bit at *position* set to *bit*.

    Examples:
        >>> update_bit(22, 3, 1)
        30
        >>> update_bit(22, 2, 0)
        18
    """
    mask = ~(1 << position)
    return (number & mask) | (bit << position)
