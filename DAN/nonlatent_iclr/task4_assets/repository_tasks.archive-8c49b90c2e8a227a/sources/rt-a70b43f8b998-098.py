def binomial_coefficient(n: int, k: int) -> int:
    """
    Since Here we Find the Binomial Coefficient:
    https://en.wikipedia.org/wiki/Binomial_coefficient
    C(n,k) = n! / k!(n-k)!
    :param n: 2 times of Number of nodes
    :param k: Number of nodes
    :return:  Integer Value

    >>> binomial_coefficient(4, 2)
    6
    """
    result = 1  # To kept the Calculated Value
    # Since C(n, k) = C(n, n-k)
    k = min(k, n - k)
    # Calculate C(n,k)
    for i in range(k):
        result *= n - i
        result //= i + 1
    return result
