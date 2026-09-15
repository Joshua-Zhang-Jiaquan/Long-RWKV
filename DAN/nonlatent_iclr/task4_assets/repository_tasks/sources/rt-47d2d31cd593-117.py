def complete_graph(vertices_number: int) -> dict:
    """
    Generate a complete graph with vertices_number vertices.
    @input: vertices_number (number of vertices),
            directed (False if the graph is undirected, True otherwise)
    @example:
    >>> complete_graph(3)
    {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    """
    return {
        i: [j for j in range(vertices_number) if i != j] for i in range(vertices_number)
    }
