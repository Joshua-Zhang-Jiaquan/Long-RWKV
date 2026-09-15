def solution():
    """
    Returns the last 10 digits of the series, 1^1 + 2^2 + 3^3 + ... + 1000^1000.

    >>> solution()
    '9110846700'
    """
    total = 0
    for i in range(1, 1001):
        total += i**i
    return str(total)[-10:]
