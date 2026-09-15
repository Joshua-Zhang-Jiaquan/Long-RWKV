def nimply_gate(input_1: int, input_2: int) -> int:
    """
    Calculate NIMPLY of the input values

    >>> nimply_gate(0, 0)
    0
    >>> nimply_gate(0, 1)
    0
    >>> nimply_gate(1, 0)
    1
    >>> nimply_gate(1, 1)
    0
    """
    return int(input_1 == 1 and input_2 == 0)
