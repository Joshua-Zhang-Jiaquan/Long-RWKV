def longest_non_repeat_v1(string: str) -> int:
    """Find the length of the longest substring without repeating characters.

    Args:
        string: Input string to search.

    Returns:
        Length of the longest non-repeating substring.

    Examples:
        >>> longest_non_repeat_v1("abcabcbb")
        3
    """
    if string is None:
        return 0
    char_index = {}
    max_length = 0
    start = 0
    for index in range(len(string)):
        if string[index] in char_index:
            start = max(char_index[string[index]], start)
        char_index[string[index]] = index + 1
        max_length = max(max_length, index - start + 1)
    return max_length
