def solution(numerator: int = 3, denominator: int = 7, limit: int = 1000000) -> int:
    """
    Returns the closest numerator of the fraction immediately to the
    left of given fraction (numerator/denominator) from a list of reduced
    proper fractions.
    >>> solution()
    428570
    >>> solution(3, 7, 8)
    2
    >>> solution(6, 7, 60)
    47
    """
    max_numerator = 0
    max_denominator = 1

    for current_denominator in range(1, limit + 1):
        current_numerator = current_denominator * numerator // denominator
        if current_denominator % denominator == 0:
            current_numerator -= 1
        if current_numerator * max_denominator > current_denominator * max_numerator:
            max_numerator = current_numerator
            max_denominator = current_denominator
    return max_numerator
