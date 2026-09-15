assert create_sample_graph() == {'A': [('B', 3), ('D', 7)], 'B': [('A', 8), ('C', 2)], 'C': [('A', 5), ('D', 1)], 'D': [('A', 2)]}
