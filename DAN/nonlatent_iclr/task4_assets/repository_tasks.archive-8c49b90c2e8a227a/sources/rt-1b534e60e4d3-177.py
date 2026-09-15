def solution(power: int = 1000) -> int:
    """Returns the sum of the digits of the number 2^power.
    >>> solution(1000)
    1366
    >>> solution(50)
    76
    >>> solution(20)
    31
    >>> solution(15)
    26
    """
    num = 2**power
    string_num = str(num)
    list_num = list(string_num)
    sum_of_num = 0

    for i in list_num:
        sum_of_num += int(i)

    return sum_of_num
