def least_common_multiple_slow(first_num: int, second_num: int) -> int:
    """
    Find the least common multiple of two numbers.

    Learn more: https://en.wikipedia.org/wiki/Least_common_multiple

    >>> least_common_multiple_slow(5, 2)
    10
    >>> least_common_multiple_slow(12, 76)
    228
    """
    max_num = first_num if first_num >= second_num else second_num
    common_mult = max_num
    while (common_mult % first_num > 0) or (common_mult % second_num > 0):
        common_mult += max_num
    return common_mult
