def binary_multiply(a: int, b: int) -> int:
    """
    Multiply 'a' and 'b' using bitwise multiplication.

    Parameters:
    a (int): The first number.
    b (int): The second number.

    Returns:
    int: a * b

    Examples:
    >>> binary_multiply(2, 3)
    6
    >>> binary_multiply(5, 0)
    0
    >>> binary_multiply(3, 4)
    12
    >>> binary_multiply(10, 5)
    50
    >>> binary_multiply(0, 5)
    0
    >>> binary_multiply(2, 1)
    2
    >>> binary_multiply(1, 10)
    10
    """
    res = 0
    while b > 0:
        if b & 1:
            res += a

        a += a
        b >>= 1

    return res
