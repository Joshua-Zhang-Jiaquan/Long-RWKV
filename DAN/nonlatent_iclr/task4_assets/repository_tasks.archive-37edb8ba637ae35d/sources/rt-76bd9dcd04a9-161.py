def solution(pence: int = 200) -> int:
    """Returns the number of different ways to make X pence using any number of coins.
    The solution is based on dynamic programming paradigm in a bottom-up fashion.

    >>> solution(500)
    6295434
    >>> solution(200)
    73682
    >>> solution(50)
    451
    >>> solution(10)
    11
    """
    coins = [1, 2, 5, 10, 20, 50, 100, 200]
    number_of_ways = [0] * (pence + 1)
    number_of_ways[0] = 1  # base case: 1 way to make 0 pence

    for coin in coins:
        for i in range(coin, pence + 1, 1):
            number_of_ways[i] += number_of_ways[i - coin]
    return number_of_ways[pence]
