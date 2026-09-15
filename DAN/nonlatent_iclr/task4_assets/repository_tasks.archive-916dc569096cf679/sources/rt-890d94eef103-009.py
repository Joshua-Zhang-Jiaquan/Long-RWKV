def get_avg(number_1: int, number_2: int) -> int:
    """
    Return the mid-number(whole) of two integers a and b

    >>> get_avg(10, 15)
    12

    >>> get_avg(20, 300)
    160

    >>> get_avg("abcd", 300)
    Traceback (most recent call last):
        ...
    TypeError: can only concatenate str (not "int") to str

    >>> get_avg(10.5,50.25)
    30
    """
    return int((number_1 + number_2) / 2)
