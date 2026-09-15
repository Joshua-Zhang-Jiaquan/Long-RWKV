def rankine_to_celsius(rankine: float, ndigits: int = 2) -> float:
    """
    Convert a given value from Rankine to Celsius and round it to 2 decimal places.
    Wikipedia reference: https://en.wikipedia.org/wiki/Rankine_scale
    Wikipedia reference: https://en.wikipedia.org/wiki/Celsius

    >>> rankine_to_celsius(273.354, 3)
    -121.287
    >>> rankine_to_celsius(273.354, 0)
    -121.0
    >>> rankine_to_celsius(273.15)
    -121.4
    >>> rankine_to_celsius(300)
    -106.48
    >>> rankine_to_celsius("315.5")
    -97.87
    >>> rankine_to_celsius("rankine")
    Traceback (most recent call last):
        ...
    ValueError: could not convert string to float: 'rankine'
    """
    return round((float(rankine) - 491.67) * 5 / 9, ndigits)
