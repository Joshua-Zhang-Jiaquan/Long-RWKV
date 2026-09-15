def equation(x: float) -> float:
    """
    >>> equation(5)
    -15
    >>> equation(0)
    10
    >>> equation(-5)
    -15
    >>> equation(0.1)
    9.99
    >>> equation(-0.1)
    9.99
    """
    return 10 - x * x
