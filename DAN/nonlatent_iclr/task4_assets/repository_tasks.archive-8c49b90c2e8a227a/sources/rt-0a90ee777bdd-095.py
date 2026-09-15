def clear_least_significant_set_bit(number: int) -> int:
    """
    Clear the least significant set bit (rightmost 1 bit).

    Subtracting 1 changes the rightmost 1 to 0 and the 0 bits to its right to 1.
    ANDing the result with the original number therefore clears that set bit.
    For negative integers, Python's infinite sign extension is used.
    https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetKernighan

    >>> clear_least_significant_set_bit(0b101100)  # 0b101000
    40
    >>> clear_least_significant_set_bit(0b1000)  # 0b0
    0
    >>> clear_least_significant_set_bit(0)
    0
    >>> clear_least_significant_set_bit(0b1111)  # 0b1110
    14
    >>> clear_least_significant_set_bit(-5)
    -6
    """
    return number & (number - 1)
