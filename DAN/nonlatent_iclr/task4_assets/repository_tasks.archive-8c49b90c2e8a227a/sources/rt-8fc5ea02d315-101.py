def digital_root(number: int) -> int:
    """
    Return the digital root of a non-negative integer number.

    The digital root is the single-digit value obtained by repeatedly summing
    the decimal digits of number until only one digit remains. The input is taken by
    its absolute value, so negative numbers behave like their positive
    counterpart.

    >>> digital_root(0)
    0
    >>> digital_root(9)
    9
    >>> digital_root(38)
    2
    >>> digital_root(12345)
    6
    >>> digital_root(-45)
    9
    >>> digital_root(999999999999)
    9
    """
    number = abs(number)
    while number >= 10:
        number = sum(int(digit) for digit in str(number))
    return number
