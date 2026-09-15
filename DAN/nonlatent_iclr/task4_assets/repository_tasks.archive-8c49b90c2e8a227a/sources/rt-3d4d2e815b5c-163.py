def count_divisors(n):
    n_divisors = 1
    i = 2
    while i * i <= n:
        multiplicity = 0
        while n % i == 0:
            n //= i
            multiplicity += 1
        n_divisors *= multiplicity + 1
        i += 1
    if n > 1:
        n_divisors *= 2
    return n_divisors
