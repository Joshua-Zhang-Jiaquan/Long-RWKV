def solution(n: int = 1000000) -> int:
    """Returns the number under n that generates the longest sequence using the
    formula:
    n → n/2 (n is even)
    n → 3n + 1 (n is odd)

    >>> solution(1000000)
    837799
    >>> solution(200)
    171
    >>> solution(5000)
    3711
    >>> solution(15000)
    13255
    """
    largest_number = 1
    pre_counter = 1
    counters = {1: 1}

    for input1 in range(2, n):
        counter = 0
        number = input1

        while True:
            if number in counters:
                counter += counters[number]
                break
            if number % 2 == 0:
                number //= 2
                counter += 1
            else:
                number = (3 * number) + 1
                counter += 1

        if input1 not in counters:
            counters[input1] = counter

        if counter > pre_counter:
            largest_number = input1
            pre_counter = counter
    return largest_number
