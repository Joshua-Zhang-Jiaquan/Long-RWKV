def abs_val(num: float) -> float:
    """
    Find the absolute value of a number.

    >>> abs_val(-5.1)
    5.1
    >>> abs_val(-5) == abs_val(5)
    True
    >>> abs_val(0)
    0
    """
    return -num if num < 0 else num
