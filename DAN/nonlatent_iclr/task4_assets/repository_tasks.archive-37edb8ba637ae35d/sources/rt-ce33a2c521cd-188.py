def solution(max_base: int = 10, max_power: int = 22) -> int:
    """
    Returns the count of all n-digit numbers which are nth power
    >>> solution(10, 22)
    49
    >>> solution(0, 0)
    0
    >>> solution(1, 1)
    0
    >>> solution(-1, -1)
    0
    """
    bases = range(1, max_base)
    powers = range(1, max_power)
    return sum(
        1 for power in powers for base in bases if len(str(base**power)) == power
    )
