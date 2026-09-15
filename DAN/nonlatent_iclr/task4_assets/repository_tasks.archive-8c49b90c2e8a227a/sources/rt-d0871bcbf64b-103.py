def fahrenheit_to_celsius(fahrenheit: float, ndigits: int = 2) -> float:
    """
    Convert a given value from Fahrenheit to Celsius and round it to 2 decimal places.
    Wikipedia reference: https://en.wikipedia.org/wiki/Fahrenheit
    Wikipedia reference: https://en.wikipedia.org/wiki/Celsius

    >>> fahrenheit_to_celsius(273.354, 3)
    134.086
    >>> fahrenheit_to_celsius(273.354, 0)
    134.0
    >>> fahrenheit_to_celsius(0)
    -17.78
    >>> fahrenheit_to_celsius(20.0)
    -6.67
    >>> fahrenheit_to_celsius(40.0)
    4.44
    >>> fahrenheit_to_celsius(60)
    15.56
    >>> fahrenheit_to_celsius(80)
    26.67
    >>> fahrenheit_to_celsius("100")
    37.78
    >>> fahrenheit_to_celsius("fahrenheit")
    Traceback (most recent call last):
        ...
    ValueError: could not convert string to float: 'fahrenheit'
    """
    return round((float(fahrenheit) - 32) * 5 / 9, ndigits)
