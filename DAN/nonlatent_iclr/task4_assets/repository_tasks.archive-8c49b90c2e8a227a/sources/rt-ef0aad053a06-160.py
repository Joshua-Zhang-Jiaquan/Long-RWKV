def reaumur_to_kelvin(reaumur: float, ndigits: int = 2) -> float:
    """
    Convert a given value from reaumur to Kelvin and round it to 2 decimal places.
    Reference:- http://www.csgnetwork.com/temp2conv.html

    >>> reaumur_to_kelvin(0)
    273.15
    >>> reaumur_to_kelvin(20.0)
    298.15
    >>> reaumur_to_kelvin(40)
    323.15
    >>> reaumur_to_kelvin("reaumur")
    Traceback (most recent call last):
        ...
    ValueError: could not convert string to float: 'reaumur'
    """
    return round((float(reaumur) * 1.25 + 273.15), ndigits)
