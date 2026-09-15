def get_longest_non_repeat_v1(string: str) -> tuple[int, str]:
    """Find the longest substring without repeating characters.

    Args:
        string: Input string to search.

    Returns:
        A tuple of (length, substring) for the longest non-repeating substring.

    Examples:
        >>> get_longest_non_repeat_v1("abcabcbb")
        (3, 'abc')
    """
    if string is None:
        return 0, ""
    substring = ""
    char_index = {}
    max_length = 0
    start = 0
    for index in range(len(string)):
        if string[index] in char_index:
            start = max(char_index[string[index]], start)
        char_index[string[index]] = index + 1
        if index - start + 1 > max_length:
            max_length = index - start + 1
            substring = string[start : index + 1]
    return max_length, substring
