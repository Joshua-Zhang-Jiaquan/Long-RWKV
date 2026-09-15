def get_bit(number: int, position: int) -> int:
    """Get the bit value at a specific position.

    Shifts 1 over by *position* bits and ANDs with *number* to isolate
    the target bit.

    Args:
        number: The integer to inspect.
        position: Zero-based bit index (0 is the least significant bit).

    Returns:
        1 if the bit at *position* is set, 0 otherwise.

    Examples:
        >>> get_bit(22, 2)
        1
        >>> get_bit(22, 3)
        0
    """
    return (number & (1 << position)) != 0
