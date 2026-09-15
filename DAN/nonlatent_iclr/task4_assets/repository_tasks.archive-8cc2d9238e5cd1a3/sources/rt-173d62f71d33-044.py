def get_longest_non_repeat_v2(string: str) -> tuple[int, str]:
    """Find the longest substring without repeating characters.

    Args:
        string: Input string to search.

    Returns:
        A tuple of (length, substring) for the longest non-repeating substring.

    Examples:
        >>> get_longest_non_repeat_v2("abcabcbb")
        (3, 'abc')
    """
    if string is None:
        return 0, ""
    substring = ""
    start, max_length = 0, 0
    used_char = {}
    for index, char in enumerate(string):
        if char in used_char and start <= used_char[char]:
            start = used_char[char] + 1
        else:
            if index - start + 1 > max_length:
                max_length = index - start + 1
                substring = string[start : index + 1]
        used_char[char] = index
    return max_length, substring
