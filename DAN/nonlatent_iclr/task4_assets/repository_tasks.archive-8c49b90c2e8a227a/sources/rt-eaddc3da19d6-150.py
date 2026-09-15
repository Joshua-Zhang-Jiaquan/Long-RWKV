def rankine_to_fahrenheit(rankine: float, ndigits: int = 2) -> float:
    """
    Convert a given value from Rankine to Fahrenheit and round it to 2 decimal places.
    Wikipedia reference: https://en.wikipedia.org/wiki/Rankine_scale
    Wikipedia reference: https://en.wikipedia.org/wiki/Fahrenheit

    >>> rankine_to_fahrenheit(273.15)
    -186.52
    >>> rankine_to_fahrenheit(300)
    -159.67
    >>> rankine_to_fahrenheit("315.5")
    -144.17
    >>> rankine_to_fahrenheit("rankine")
    Traceback (most recent call last):
        ...
    ValueError: could not convert string to float: 'rankine'
    """
    return round(float(rankine) - 459.67, ndigits)
