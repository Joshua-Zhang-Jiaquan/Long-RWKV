def sum_32(a: int, b: int) -> int:
    """
    Add two numbers as 32-bit ints.

    Arguments:
        a {[int]} -- [first given int]
        b {[int]} -- [second given int]

    Returns:
        (a + b) as an unsigned 32-bit int

    >>> sum_32(1, 1)
    2
    >>> sum_32(2, 3)
    5
    >>> sum_32(0, 0)
    0
    >>> sum_32(-1, -1)
    4294967294
    >>> sum_32(4294967295, 1)
    0
    """
    return (a + b) % 2**32
