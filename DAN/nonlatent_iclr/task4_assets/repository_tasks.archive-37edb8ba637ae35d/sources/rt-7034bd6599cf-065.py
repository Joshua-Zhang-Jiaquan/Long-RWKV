def swap(a: int, b: int) -> tuple[int, int]:
    """
    Return a tuple (b, a) when given two integers a and b
    >>> swap(2,3)
    (3, 2)
    >>> swap(3,4)
    (4, 3)
    >>> swap(67, 12)
    (12, 67)
    >>> swap(3,-4)
    (-4, 3)
    """
    a ^= b
    b ^= a
    a ^= b
    return a, b
