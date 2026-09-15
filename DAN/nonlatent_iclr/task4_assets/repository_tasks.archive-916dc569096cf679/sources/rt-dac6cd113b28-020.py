def djb2(s: str) -> int:
    """
    Implementation of djb2 hash algorithm that
    is popular because of it's magic constants.

    >>> djb2('Algorithms')
    3782405311

    >>> djb2('scramble bits')
    1609059040
    """
    hash_value = 5381
    for x in s:
        hash_value = ((hash_value << 5) + hash_value) + ord(x)
    return hash_value & 0xFFFFFFFF
