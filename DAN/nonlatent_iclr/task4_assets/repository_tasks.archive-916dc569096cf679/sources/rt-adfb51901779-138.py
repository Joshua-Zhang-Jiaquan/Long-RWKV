def clear_bit(number: int, position: int) -> int:
    """Clear the bit at a specific position to 0.

    Creates a mask with all bits set except at *position*, then ANDs
    with *number*.

    Args:
        number: The integer to modify.
        position: Zero-based bit index to clear.

    Returns:
        The integer with the bit at *position* cleared to 0.

    Examples:
        >>> clear_bit(22, 2)
        18
    """
    mask = ~(1 << position)
    return number & mask
