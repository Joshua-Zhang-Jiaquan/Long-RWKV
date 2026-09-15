def kelvin_to_fahrenheit(kelvin: float, ndigits: int = 2) -> float:
    """
    Convert a given value from Kelvin to Fahrenheit and round it to 2 decimal places.
    Wikipedia reference: https://en.wikipedia.org/wiki/Kelvin
    Wikipedia reference: https://en.wikipedia.org/wiki/Fahrenheit

    >>> kelvin_to_fahrenheit(273.354, 3)
    32.367
    >>> kelvin_to_fahrenheit(273.354, 0)
    32.0
    >>> kelvin_to_fahrenheit(273.15)
    32.0
    >>> kelvin_to_fahrenheit(300)
    80.33
    >>> kelvin_to_fahrenheit("315.5")
    108.23
    >>> kelvin_to_fahrenheit("kelvin")
    Traceback (most recent call last):
        ...
    ValueError: could not convert string to float: 'kelvin'
    """
    return round(((float(kelvin) - 273.15) * 9 / 5) + 32, ndigits)
