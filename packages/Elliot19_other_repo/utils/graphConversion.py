
def __conversion__(G):
    newGraph = [{} for x  in range(len(G))]
    for x in G:
        for y in G[x]:
            newGraph[x][y]=G.adj[x][y]['weight']
    return newGraph
