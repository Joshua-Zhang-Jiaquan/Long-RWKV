def is_for_table(string1: str, string2: str, count: int) -> bool:
    """
    >>> is_for_table('__1','011',2)
    True

    >>> is_for_table('01_','001',1)
    False
    """
    list1 = list(string1)
    list2 = list(string2)
    count_n = sum(item1 != item2 for item1, item2 in zip(list1, list2))
    return count_n == count
