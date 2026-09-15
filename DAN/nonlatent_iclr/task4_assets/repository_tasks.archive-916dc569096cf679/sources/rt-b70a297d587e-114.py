def fahrenheit_to_rankine(fahrenheit: float, ndigits: int = 2) -> float:
    """
    Convert a given value from Fahrenheit to Rankine and round it to 2 decimal places.
    Wikipedia reference: https://en.wikipedia.org/wiki/Fahrenheit
    Wikipedia reference: https://en.wikipedia.org/wiki/Rankine_scale

    >>> fahrenheit_to_rankine(273.354, 3)
    733.024
    >>> fahrenheit_to_rankine(273.354, 0)
    733.0
    >>> fahrenheit_to_rankine(0)
    459.67
    >>> fahrenheit_to_rankine(20.0)
    479.67
    >>> fahrenheit_to_rankine(40.0)
    499.67
    >>> fahrenheit_to_rankine(60)
    519.67
    >>> fahrenheit_to_rankine(80)
    539.67
    >>> fahrenheit_to_rankine("100")
    559.67
    >>> fahrenheit_to_rankine("fahrenheit")
    Traceback (most recent call last):
        ...
    ValueError: could not convert string to float: 'fahrenheit'
    """
    return round(float(fahrenheit) + 459.67, ndigits)
