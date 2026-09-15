def clear_bit(number: int, position: int) -> int:
    """
    Set the bit at position to 0.

    Details: perform bitwise and for given number and X.
    Where X is a number with all the bits - ones and bit on given
    position - zero.

    >>> clear_bit(0b10010, 1) # 0b10000
    16
    >>> clear_bit(0b0, 5) # 0b0
    0
    """
    return number & ~(1 << position)
