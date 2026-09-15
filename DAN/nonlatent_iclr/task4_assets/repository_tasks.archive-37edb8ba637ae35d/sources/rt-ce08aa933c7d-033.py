def set_bit(number: int, position: int) -> int:
    """
    Set the bit at position to 1.

    Details: perform bitwise or for given number and X.
    Where X is a number with all the bits - zeroes and bit on given
    position - one.

    >>> set_bit(0b1101, 1) # 0b1111
    15
    >>> set_bit(0b0, 5) # 0b100000
    32
    >>> set_bit(0b1111, 1) # 0b1111
    15
    """
    return number | (1 << position)
