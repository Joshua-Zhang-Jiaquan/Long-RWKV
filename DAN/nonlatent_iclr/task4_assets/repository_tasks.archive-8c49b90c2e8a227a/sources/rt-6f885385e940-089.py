def rotate_v3(array: list[int] | None, k: int) -> list[int] | None:
    """Rotate array to the right by k steps using slicing.

    Args:
        array: List of integers to rotate, or None.
        k: Number of positions to rotate right.

    Returns:
        New rotated list, or None if input is None.

    Examples:
        >>> rotate_v3([1, 2, 3, 4, 5, 6, 7], 3)
        [5, 6, 7, 1, 2, 3, 4]
    """
    if array is None:
        return None
    length = len(array)
    k = k % length
    return array[length - k :] + array[: length - k]
