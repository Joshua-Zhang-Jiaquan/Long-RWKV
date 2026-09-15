def sum_of_digits_compact(n: int) -> int:
    """
    Find the sum of digits of a number
    >>> sum_of_digits_compact(12345)
    15
    >>> sum_of_digits_compact(123)
    6
    >>> sum_of_digits_compact(-123)
    6
    >>> sum_of_digits_compact(0)
    0
    """
    return sum(int(c) for c in str(abs(n)))
