def is_even_using_shift_operator(number: int) -> bool:
    """
    Returns True if the input integer is even.

    Explanation:
    In binary, even numbers end with 0, odd numbers end with 1.
    Examples:
    2  -> 10
    3  -> 11
    4  -> 100
    5  -> 101

    For odd numbers, the last bit is always 1.
    Using shift:
    (n >> 1) << 1 removes the last bit.
    If result equals n, n is even.

    >>> is_even_using_shift_operator(1)
    False
    >>> is_even_using_shift_operator(4)
    True
    >>> is_even_using_shift_operator(9)
    False
    >>> is_even_using_shift_operator(15)
    False
    >>> is_even_using_shift_operator(40)
    True
    >>> is_even_using_shift_operator(100)
    True
    >>> is_even_using_shift_operator(101)
    False
    """
    return (number >> 1) << 1 == number
