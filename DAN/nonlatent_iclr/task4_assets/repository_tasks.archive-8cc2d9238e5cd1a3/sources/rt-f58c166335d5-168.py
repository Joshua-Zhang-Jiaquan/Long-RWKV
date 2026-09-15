def count_divisors(n):
    divisors_count = 1
    i = 2
    while i * i <= n:
        multiplicity = 0
        while n % i == 0:
            n //= i
            multiplicity += 1
        divisors_count *= multiplicity + 1
        i += 1
    if n > 1:
        divisors_count *= 2
    return divisors_count
