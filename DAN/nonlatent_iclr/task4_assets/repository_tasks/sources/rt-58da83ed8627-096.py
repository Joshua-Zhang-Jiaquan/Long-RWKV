def xnor_gate(input_1: int, input_2: int) -> int:
    """
    Calculate XOR of the input values
    >>> xnor_gate(0, 0)
    1
    >>> xnor_gate(0, 1)
    0
    >>> xnor_gate(1, 0)
    0
    >>> xnor_gate(1, 1)
    1
    """
    return 1 if input_1 == input_2 else 0
