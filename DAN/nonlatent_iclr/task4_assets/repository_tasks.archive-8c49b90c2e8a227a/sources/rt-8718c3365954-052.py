def sdbm(plain_text: str) -> int:
    """
    Function implements sdbm hash, easy to use, great for bits scrambling.
    iterates over each character in the given string and applies function to each of
    them.

    >>> sdbm('Algorithms')
    1462174910723540325254304520539387479031000036

    >>> sdbm('scramble bits')
    730247649148944819640658295400555317318720608290373040936089
    """
    hash_value = 0
    for plain_chr in plain_text:
        hash_value = (
            ord(plain_chr) + (hash_value << 6) + (hash_value << 16) - hash_value
        )
    return hash_value
