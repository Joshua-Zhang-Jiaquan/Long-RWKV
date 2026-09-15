def contains_an_even_digit(n: int) -> bool:
    """
    Return True if n contains an even digit.
    >>> contains_an_even_digit(0)
    True
    >>> contains_an_even_digit(975317933)
    False
    >>> contains_an_even_digit(-245679)
    True
    """
    return any(digit in "02468" for digit in str(n))
