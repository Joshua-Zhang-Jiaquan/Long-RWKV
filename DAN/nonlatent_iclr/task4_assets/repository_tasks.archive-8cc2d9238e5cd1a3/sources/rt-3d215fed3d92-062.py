def nor_gate(input_1: int, input_2: int) -> int:
    """
    >>> nor_gate(0, 0)
    1
    >>> nor_gate(0, 1)
    0
    >>> nor_gate(1, 0)
    0
    >>> nor_gate(1, 1)
    0
    >>> nor_gate(0.0, 0.0)
    1
    >>> nor_gate(0, -7)
    0
    """
    return int(input_1 == input_2 == 0)
