def fahrenheit_to_kelvin(fahrenheit: float, ndigits: int = 2) -> float:
    """
    Convert a given value from Fahrenheit to Kelvin and round it to 2 decimal places.
    Wikipedia reference: https://en.wikipedia.org/wiki/Fahrenheit
    Wikipedia reference: https://en.wikipedia.org/wiki/Kelvin

    >>> fahrenheit_to_kelvin(273.354, 3)
    407.236
    >>> fahrenheit_to_kelvin(273.354, 0)
    407.0
    >>> fahrenheit_to_kelvin(0)
    255.37
    >>> fahrenheit_to_kelvin(20.0)
    266.48
    >>> fahrenheit_to_kelvin(40.0)
    277.59
    >>> fahrenheit_to_kelvin(60)
    288.71
    >>> fahrenheit_to_kelvin(80)
    299.82
    >>> fahrenheit_to_kelvin("100")
    310.93
    >>> fahrenheit_to_kelvin("fahrenheit")
    Traceback (most recent call last):
        ...
    ValueError: could not convert string to float: 'fahrenheit'
    """
    return round(((float(fahrenheit) - 32) * 5 / 9) + 273.15, ndigits)
