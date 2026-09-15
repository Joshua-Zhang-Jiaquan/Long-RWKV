assert create_negative_cycle_graph() == {'A': [('B', 1)], 'B': [('C', -3)], 'C': [('D', 2)], 'D': [('B', -1)]}
