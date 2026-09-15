def max_ones_index(array: list[int]) -> int:
    """Find the index of 0 to replace with 1 for the longest run of 1s.

    Args:
        array: Binary array containing only 0s and 1s.

    Returns:
        Index of the 0 to flip, or -1 if no 0 exists.

    Examples:
        >>> max_ones_index([1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1])
        3
    """
    length = len(array)
    max_count = 0
    max_index = 0
    prev_zero = -1
    prev_prev_zero = -1

    for current in range(length):
        if array[current] == 0:
            if current - prev_prev_zero > max_count:
                max_count = current - prev_prev_zero
                max_index = prev_zero

            prev_prev_zero = prev_zero
            prev_zero = current

    if length - prev_prev_zero > max_count:
        max_index = prev_zero

    return max_index
