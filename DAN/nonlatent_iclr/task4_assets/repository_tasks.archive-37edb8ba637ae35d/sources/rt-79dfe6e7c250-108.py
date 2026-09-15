def set_bit(number: int, position: int) -> int:
    """Set the bit at a specific position to 1.

    Shifts 1 over by *position* bits and ORs with *number* so that only
    the bit at *position* is turned on.

    Args:
        number: The integer to modify.
        position: Zero-based bit index to set.

    Returns:
        The integer with the bit at *position* set to 1.

    Examples:
        >>> set_bit(22, 3)
        30
    """
    return number | (1 << position)
