def create_sparse(max_node: int, parent: list[list[int]]) -> list[list[int]]:
    """
    creating sparse table which saves each nodes 2^i-th parent
    >>> max_node = 6
    >>> parent = [[0, 0, 1, 1, 2, 2, 3]] + [[0] * 7 for _ in range(19)]
    >>> parent = create_sparse(max_node=max_node, parent=parent)
    >>> parent[0]
    [0, 0, 1, 1, 2, 2, 3]
    >>> parent[1]
    [0, 0, 0, 0, 1, 1, 1]
    >>> parent[2]
    [0, 0, 0, 0, 0, 0, 0]

    >>> max_node = 1
    >>> parent = [[0, 0]] + [[0] * 2 for _ in range(19)]
    >>> parent = create_sparse(max_node=max_node, parent=parent)
    >>> parent[0]
    [0, 0]
    >>> parent[1]
    [0, 0]
    """
    j = 1
    while (1 << j) < max_node:
        for i in range(1, max_node + 1):
            parent[j][i] = parent[j - 1][parent[j - 1][i]]
        j += 1
    return parent
