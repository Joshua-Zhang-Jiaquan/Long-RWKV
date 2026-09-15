def solution(power: int = 1000) -> int:
    """Returns the sum of the digits of the number 2^power.

    >>> solution(1000)
    1366
    >>> solution(50)
    76
    >>> solution(20)
    31
    >>> solution(15)
    26
    """
    n = 2**power
    r = 0
    while n:
        r, n = r + n % 10, n // 10
    return r
