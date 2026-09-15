def find_negative_index(array: list[int]) -> int:
    """
    Find the smallest negative index

    >>> find_negative_index([0,0,0,0])
    4
    >>> find_negative_index([4,3,2,-1])
    3
    >>> find_negative_index([1,0,-1,-10])
    2
    >>> find_negative_index([0,0,0,-1])
    3
    >>> find_negative_index([11,8,7,-3,-5,-9])
    3
    >>> find_negative_index([-1,-1,-2,-3])
    0
    >>> find_negative_index([5,1,0])
    3
    >>> find_negative_index([-5,-5,-5])
    0
    >>> find_negative_index([0])
    1
    >>> find_negative_index([])
    0
    """
    left = 0
    right = len(array) - 1

    # Edge cases such as no values or all numbers are negative.
    if not array or array[0] < 0:
        return 0

    while right + 1 > left:
        mid = (left + right) // 2
        num = array[mid]

        # Num must be negative and the index must be greater than or equal to 0.
        if num < 0 and array[mid - 1] >= 0:
            return mid

        if num >= 0:
            left = mid + 1
        else:
            right = mid - 1
    # No negative numbers so return the last index of the array + 1 which is the length.
    return len(array)
