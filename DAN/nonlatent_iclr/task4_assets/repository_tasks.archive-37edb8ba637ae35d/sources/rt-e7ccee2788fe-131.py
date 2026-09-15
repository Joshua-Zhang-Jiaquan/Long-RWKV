def reaumur_to_rankine(reaumur: float, ndigits: int = 2) -> float:
    """
    Convert a given value from reaumur to rankine and round it to 2 decimal places.
    Reference:- http://www.csgnetwork.com/temp2conv.html

    >>> reaumur_to_rankine(0)
    491.67
    >>> reaumur_to_rankine(20.0)
    536.67
    >>> reaumur_to_rankine(40)
    581.67
    >>> reaumur_to_rankine("reaumur")
    Traceback (most recent call last):
        ...
    ValueError: could not convert string to float: 'reaumur'
    """
    return round((float(reaumur) * 2.25 + 32 + 459.67), ndigits)
