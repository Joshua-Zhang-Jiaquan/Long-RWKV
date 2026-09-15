def solution(n: int = 1000) -> int:
    """Returns the index of the first term in the Fibonacci sequence to contain
    n digits.

    >>> solution(1000)
    4782
    >>> solution(100)
    476
    >>> solution(50)
    237
    >>> solution(3)
    12
    """
    f1, f2 = 1, 1
    index = 2
    while True:
        i = 0
        f = f1 + f2
        f1, f2 = f2, f
        index += 1
        for _ in str(f):
            i += 1
        if i == n:
            break
    return index
