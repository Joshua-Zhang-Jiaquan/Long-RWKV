def sieve_of_eratosthenes(n):
    """
    function to find and print prime numbers up
    to the specified number

    :param n: upper limit for finding all primes less than this value
    """
    primes = [True] * (n + 1)
    # because p is the smallest prime
    p = 2

    while p * p <= n:
        # if p is not marked as False, it is a prime
        if primes[p]:
            # mark all the multiples of number as False
            for i in range(p * 2, n + 1, p):
                primes[i] = False
        p += 1

    # getting all primes
    primes = [element for element in range(2, n + 1) if primes[element]]

    return primes
