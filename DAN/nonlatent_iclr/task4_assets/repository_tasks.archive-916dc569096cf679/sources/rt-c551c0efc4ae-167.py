def reaumur_to_celsius(reaumur: float, ndigits: int = 2) -> float:
    """
    Convert a given value from reaumur to celsius and round it to 2 decimal places.
    Reference:- http://www.csgnetwork.com/temp2conv.html

    >>> reaumur_to_celsius(0)
    0.0
    >>> reaumur_to_celsius(20.0)
    25.0
    >>> reaumur_to_celsius(40)
    50.0
    >>> reaumur_to_celsius("reaumur")
    Traceback (most recent call last):
        ...
    ValueError: could not convert string to float: 'reaumur'
    """
    return round((float(reaumur) * 1.25), ndigits)
