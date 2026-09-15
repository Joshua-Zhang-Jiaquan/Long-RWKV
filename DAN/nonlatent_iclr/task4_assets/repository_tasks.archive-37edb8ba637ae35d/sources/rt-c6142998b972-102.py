def kelvin_to_celsius(kelvin: float, ndigits: int = 2) -> float:
    """
    Convert a given value from Kelvin to Celsius and round it to 2 decimal places.
    Wikipedia reference: https://en.wikipedia.org/wiki/Kelvin
    Wikipedia reference: https://en.wikipedia.org/wiki/Celsius

    >>> kelvin_to_celsius(273.354, 3)
    0.204
    >>> kelvin_to_celsius(273.354, 0)
    0.0
    >>> kelvin_to_celsius(273.15)
    0.0
    >>> kelvin_to_celsius(300)
    26.85
    >>> kelvin_to_celsius("315.5")
    42.35
    >>> kelvin_to_celsius("kelvin")
    Traceback (most recent call last):
        ...
    ValueError: could not convert string to float: 'kelvin'
    """
    return round(float(kelvin) - 273.15, ndigits)
