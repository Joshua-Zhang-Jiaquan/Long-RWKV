def rotate_v1(array: list[int], k: int) -> list[int]:
    """Rotate array to the right by k steps using repeated single shifts.

    Args:
        array: List of integers to rotate.
        k: Number of positions to rotate right.

    Returns:
        New rotated list.

    Examples:
        >>> rotate_v1([1, 2, 3, 4, 5, 6, 7], 3)
        [5, 6, 7, 1, 2, 3, 4]
    """
    array = array[:]
    length = len(array)
    for _ in range(k):
        temp = array[length - 1]
        for position in range(length - 1, 0, -1):
            array[position] = array[position - 1]
        array[0] = temp
    return array
