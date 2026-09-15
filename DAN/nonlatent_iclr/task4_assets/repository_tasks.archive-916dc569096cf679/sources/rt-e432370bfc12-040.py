def median_of_five(arr: list) -> int:
    """
    Return the median of the input list
    :param arr: Array to find median of
    :return: median of arr

    >>> median_of_five([2, 4, 5, 7, 899])
    5
    >>> median_of_five([5, 7, 899, 54, 32])
    32
    >>> median_of_five([5, 4, 3, 2])
    4
    >>> median_of_five([3, 5, 7, 10, 2])
    5
    """
    arr = sorted(arr)
    return arr[len(arr) // 2]
