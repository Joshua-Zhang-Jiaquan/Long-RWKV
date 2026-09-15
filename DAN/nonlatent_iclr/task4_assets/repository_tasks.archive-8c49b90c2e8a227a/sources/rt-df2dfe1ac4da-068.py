def binary_mod_multiply(a: int, b: int, modulus: int) -> int:
    """
    Calculate (a * b) % c using binary multiplication and modular arithmetic.

    Parameters:
    a (int): The first number.
    b (int): The second number.
    modulus (int): The modulus.

    Returns:
    int: (a * b) % modulus.

    Examples:
    >>> binary_mod_multiply(2, 3, 5)
    1
    >>> binary_mod_multiply(5, 0, 7)
    0
    >>> binary_mod_multiply(3, 4, 6)
    0
    >>> binary_mod_multiply(10, 5, 13)
    11
    >>> binary_mod_multiply(2, 1, 5)
    2
    >>> binary_mod_multiply(1, 10, 3)
    1
    """
    res = 0
    while b > 0:
        if b & 1:
            res = ((res % modulus) + (a % modulus)) % modulus

        a += a
        b >>= 1

    return res
