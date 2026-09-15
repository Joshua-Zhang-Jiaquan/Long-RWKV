def count_permutations(n, r=None):
    """
    Count the number of permutations of n items taken r at a time
    
    :param n: total number of items
    :param r: number of items to choose (defaults to n)
    :return: number of permutations
    """
    if r is None:
        r = n
    
    if r > n or r < 0:
        return 0
    
    if r == 0:
        return 1
    
    result = 1
    for i in range(n, n - r, -1):
        result *= i
    
    return result
