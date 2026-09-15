def sum_reverse(n: int) -> int:
    """
    Returns the sum of n and reverse of n.
    >>> sum_reverse(123)
    444
    >>> sum_reverse(3478)
    12221
    >>> sum_reverse(12)
    33
    """
    return int(n) + int(str(n)[::-1])
