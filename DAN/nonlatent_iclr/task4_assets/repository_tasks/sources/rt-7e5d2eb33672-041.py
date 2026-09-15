def insertion_sort(array: list, start: int = 0, end: int = 0) -> list:
    """
    >>> array = [4, 2, 6, 8, 1, 7, 8, 22, 14, 56, 27, 79, 23, 45, 14, 12]
    >>> insertion_sort(array, 0, len(array))
    [1, 2, 4, 6, 7, 8, 8, 12, 14, 14, 22, 23, 27, 45, 56, 79]
    >>> array = [21, 15, 11, 45, -2, -11, 46]
    >>> insertion_sort(array, 0, len(array))
    [-11, -2, 11, 15, 21, 45, 46]
    >>> array = [-2, 0, 89, 11, 48, 79, 12]
    >>> insertion_sort(array, 0, len(array))
    [-2, 0, 11, 12, 48, 79, 89]
    >>> array = ['a', 'z', 'd', 'p', 'v', 'l', 'o', 'o']
    >>> insertion_sort(array, 0, len(array))
    ['a', 'd', 'l', 'o', 'o', 'p', 'v', 'z']
    >>> array = [73.568, 73.56, -45.03, 1.7, 0, 89.45]
    >>> insertion_sort(array, 0, len(array))
    [-45.03, 0, 1.7, 73.56, 73.568, 89.45]
    """
    end = end or len(array)
    for i in range(start, end):
        temp_index = i
        temp_index_value = array[i]
        while temp_index != start and temp_index_value < array[temp_index - 1]:
            array[temp_index] = array[temp_index - 1]
            temp_index -= 1
        array[temp_index] = temp_index_value
    return array
