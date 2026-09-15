def next_permutation(arr):
    """
    Generate the next lexicographically greater permutation
    
    :param arr: current permutation
    :return: next permutation or None if current is the last
    """
    if not arr or len(arr) <= 1:
        return None
    
    arr_copy = arr[:]
    
    # Find the largest index i such that arr[i] < arr[i + 1]
    i = len(arr_copy) - 2
    while i >= 0 and arr_copy[i] >= arr_copy[i + 1]:
        i -= 1
    
    # If no such index exists, this is the last permutation
    if i == -1:
        return None
    
    # Find the largest index j such that arr[i] < arr[j]
    j = len(arr_copy) - 1
    while arr_copy[j] <= arr_copy[i]:
        j -= 1
    
    # Swap arr[i] and arr[j]
    arr_copy[i], arr_copy[j] = arr_copy[j], arr_copy[i]
    
    # Reverse the suffix starting at arr[i + 1]
    arr_copy[i + 1:] = reversed(arr_copy[i + 1:])
    
    return arr_copy
