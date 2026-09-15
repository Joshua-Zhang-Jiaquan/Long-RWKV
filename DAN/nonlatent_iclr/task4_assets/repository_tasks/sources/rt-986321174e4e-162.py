def reaumur_to_fahrenheit(reaumur: float, ndigits: int = 2) -> float:
    """
    Convert a given value from reaumur to fahrenheit and round it to 2 decimal places.
    Reference:- http://www.csgnetwork.com/temp2conv.html

    >>> reaumur_to_fahrenheit(0)
    32.0
    >>> reaumur_to_fahrenheit(20.0)
    77.0
    >>> reaumur_to_fahrenheit(40)
    122.0
    >>> reaumur_to_fahrenheit("reaumur")
    Traceback (most recent call last):
        ...
    ValueError: could not convert string to float: 'reaumur'
    """
    return round((float(reaumur) * 2.25 + 32), ndigits)
