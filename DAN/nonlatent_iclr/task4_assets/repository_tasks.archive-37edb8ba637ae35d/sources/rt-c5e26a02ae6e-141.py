def simple_fibonacci(n: int, f1: int, f2: int) -> int:
    """
    Returns the nth number of the Fibonacci sequence that
    starts with f1 and f2
    Uses the definition
    >>> simple_fibonacci(1, 5, 6)
    5
    >>> simple_fibonacci(2, 10, 11)
    11
    >>> simple_fibonacci(13, 0, 1)
    144
    >>> simple_fibonacci(10, 5, 9)
    411
    >>> simple_fibonacci(9, 2, 3)
    89
    """
    # Trivial Cases
    if n == 1:
        return f1
    elif n == 2:
        return f2

    n -= 2

    while n > 0:
        f2, f1 = f1 + f2, f2
        n -= 1

    return f2
