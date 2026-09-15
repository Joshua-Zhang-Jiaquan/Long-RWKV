def imply_gate(input_1: int, input_2: int) -> int:
    """
    Calculate IMPLY of the input values

    >>> imply_gate(0, 0)
    1
    >>> imply_gate(0, 1)
    1
    >>> imply_gate(1, 0)
    0
    >>> imply_gate(1, 1)
    1
    """
    return int(input_1 == 0 or input_2 == 1)
