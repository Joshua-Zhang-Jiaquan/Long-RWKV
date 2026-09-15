def get_bit(number: int, position: int) -> int:
    """
    Get the bit at the given position

    Details: perform bitwise and for the given number and X,
    Where X is a number with all the bits - zeroes and bit on given position - one.
    If the result is not equal to 0, then the bit on the given position is 1, else 0.

    >>> get_bit(0b1010, 0)
    0
    >>> get_bit(0b1010, 1)
    1
    >>> get_bit(0b1010, 2)
    0
    >>> get_bit(0b1010, 3)
    1
    """
    return int((number & (1 << position)) != 0)
