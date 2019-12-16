
def personalRank(G, alpha, root, max_step):
    rank = {x :0 for x in G.keys()}
    rank[root] = 1
    for k in range(max_step):
        tmp = {x: 0 for x in G.keys()}
        for i, ri in G.items():
            for j, wij in ri.items():
                tmp[j] += alpha * rank[i] / (1.0 * len(ri))
        tmp[root] += (1 - alpha)
        rank = tmp
    return rank

def test():
    G = {'A': {'a': 1, 'c': 1},
         'B': {'a': 1, 'b': 1, 'c': 1, 'd': 1},
         'C': {'c': 1, 'd': 1},
         'a': {'A': 1, 'B': 1},
         'b': {'B': 1},
         'c': {'A': 1, 'B': 1, 'C': 1},
         'd': {'B': 1, 'C': 1}}
    personalRank(G, 0.85, 'A', 100)

test()
