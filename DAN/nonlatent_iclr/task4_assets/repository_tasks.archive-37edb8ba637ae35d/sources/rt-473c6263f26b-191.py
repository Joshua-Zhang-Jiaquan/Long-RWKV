def sum_digits(num: int) -> int:
    """
    Returns the sum of every digit in num.

    >>> sum_digits(1)
    1
    >>> sum_digits(12345)
    15
    >>> sum_digits(999001)
    28
    """
    digit_sum = 0
    while num > 0:
        digit_sum += num % 10
        num //= 10
    return digit_sum
