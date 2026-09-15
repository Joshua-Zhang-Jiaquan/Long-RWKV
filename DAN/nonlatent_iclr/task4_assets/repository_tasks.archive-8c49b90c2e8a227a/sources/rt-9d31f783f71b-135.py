def kelvin_to_rankine(kelvin: float, ndigits: int = 2) -> float:
    """
    Convert a given value from Kelvin to Rankine and round it to 2 decimal places.
    Wikipedia reference: https://en.wikipedia.org/wiki/Kelvin
    Wikipedia reference: https://en.wikipedia.org/wiki/Rankine_scale

    >>> kelvin_to_rankine(273.354, 3)
    492.037
    >>> kelvin_to_rankine(273.354, 0)
    492.0
    >>> kelvin_to_rankine(0)
    0.0
    >>> kelvin_to_rankine(20.0)
    36.0
    >>> kelvin_to_rankine("40")
    72.0
    >>> kelvin_to_rankine("kelvin")
    Traceback (most recent call last):
        ...
    ValueError: could not convert string to float: 'kelvin'
    """
    return round((float(kelvin) * 9 / 5), ndigits)
