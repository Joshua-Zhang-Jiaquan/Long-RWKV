def rankine_to_kelvin(rankine: float, ndigits: int = 2) -> float:
    """
    Convert a given value from Rankine to Kelvin and round it to 2 decimal places.
    Wikipedia reference: https://en.wikipedia.org/wiki/Rankine_scale
    Wikipedia reference: https://en.wikipedia.org/wiki/Kelvin

    >>> rankine_to_kelvin(0)
    0.0
    >>> rankine_to_kelvin(20.0)
    11.11
    >>> rankine_to_kelvin("40")
    22.22
    >>> rankine_to_kelvin("rankine")
    Traceback (most recent call last):
        ...
    ValueError: could not convert string to float: 'rankine'
    """
    return round((float(rankine) * 5 / 9), ndigits)
