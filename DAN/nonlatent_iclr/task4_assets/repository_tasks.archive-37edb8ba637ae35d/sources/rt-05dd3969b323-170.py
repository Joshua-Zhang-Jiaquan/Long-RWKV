def is_9_pandigital(n: int) -> bool:
    """
    Checks whether n is a 9-digit 1 to 9 pandigital number.
    >>> is_9_pandigital(12345)
    False
    >>> is_9_pandigital(156284973)
    True
    >>> is_9_pandigital(1562849733)
    False
    """
    s = str(n)
    return len(s) == 9 and set(s) == set("123456789")
