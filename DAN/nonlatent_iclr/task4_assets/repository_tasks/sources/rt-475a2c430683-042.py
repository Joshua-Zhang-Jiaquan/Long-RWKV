def cross(items_a, items_b):
    """
    Cross product of elements in A and elements in B.

    >>> cross('AB', '12')
    ['A1', 'A2', 'B1', 'B2']
    >>> cross('ABC', '123')
    ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']
    >>> cross('ABC', '1234')
    ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4', 'C1', 'C2', 'C3', 'C4']
    >>> cross('', '12')
    []
    >>> cross('A', '')
    []
    >>> cross('', '')
    []
    """
    return [a + b for a in items_a for b in items_b]
