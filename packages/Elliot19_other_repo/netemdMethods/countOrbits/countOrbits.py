import networkx as nx
import itertools as it

## ordering as in milo 2004
def motifCounterFastProduct(G):
    if not nx.is_directed(G):
        print 'Not defined for undirected graphs'
        assert(False)
    result=[[0 for x in range(13)] for y in range(len(G))]
    for x in G:
        pre=set(G.predecessors(x))
        suc=set(G.successors(x))
        both=set(pre).intersection(suc)
        pre=pre.difference(both)
        suc=suc.difference(both)
        pre0=[y for y in pre if y<x]
        pre1=[y for y in pre if y>x]
        suc0=[y for y in suc if y<x]
        suc1=[y for y in suc if y>x]
        both0=[y for y in both if y<x]
        both1=[y for y in both if y>x]
        for n1,n2 in it.combinations(pre0,2):
            w1=G.edge[n1][x]['weight']
            w2=G.edge[n2][x]['weight']
            c1=G.has_edge(n1,n2)
            c2=G.has_edge(n2,n1)
            if (c1 and c2):
                w3=G.edge[n1][n2]['weight']
                w4=G.edge[n2][n1]['weight']
                result[x][9]+=w1*w2*w3*w4
                result[n1][9]+=w1*w2*w3*w4
                result[n2][9]+=w1*w2*w3*w4
            elif (c1):
                w3=G.edge[n1][n2]['weight']
                result[x][6]+=w1*w2*w3
                result[n1][6]+=w1*w2*w3
                result[n2][6]+=w1*w2*w3
            elif (c2):
                w3=G.edge[n2][n1]['weight']
                result[x][6]+=w1*w2*w3
                result[n1][6]+=w1*w2*w3
                result[n2][6]+=w1*w2*w3
        for n1,n2 in it.combinations(pre,2):
            w1=G.edge[n1][x]['weight']
            w2=G.edge[n2][x]['weight']
            c1=G.has_edge(n1,n2)
            c2=G.has_edge(n2,n1)
            if (not c1) and (not c2):
                result[x][1]+=w1*w2
                result[n1][1]+=w1*w2
                result[n2][1]+=w1*w2
        for n1,n2 in it.combinations(suc0,2):
            w1=G.edge[x][n1]['weight']
            w2=G.edge[x][n2]['weight']
            c1=G.has_edge(n1,n2)
            c2=G.has_edge(n2,n1)
            if (c1 and c2):
                w3=G.edge[n1][n2]['weight']
                w4=G.edge[n2][n1]['weight']
                result[x][8] +=w1*w2*w3*w4
                result[n1][8]+=w1*w2*w3*w4
                result[n2][8]+=w1*w2*w3*w4
            elif (c1):
                w3=G.edge[n1][n2]['weight']
                result[x][6] +=w1*w2*w3
                result[n1][6]+=w1*w2*w3
                result[n2][6]+=w1*w2*w3
            elif (c2):
                w3=G.edge[n2][n1]['weight']
                result[x][6] +=w1*w2*w3
                result[n1][6]+=w1*w2*w3
                result[n2][6]+=w1*w2*w3
        for n1,n2 in it.combinations(suc,2):
            w1=G.edge[x][n1]['weight']
            w2=G.edge[x][n2]['weight']
            c1=G.has_edge(n1,n2)
            c2=G.has_edge(n2,n1)
            if (not c1) and (not c2):
                result[x][0] +=w1*w2
                result[n1][0]+=w1*w2
                result[n2][0]+=w1*w2
        for n1,n2 in it.combinations(both0,2):
            w1=G.edge[x][n1]['weight']
            w2=G.edge[x][n2]['weight']
            w1*=G.edge[n1][x]['weight']
            w2*=G.edge[n2][x]['weight']
            c1=G.has_edge(n1,n2)
            c2=G.has_edge(n2,n1)
            if (c1 and c2):
                w3=G.edge[n1][n2]['weight']
                w4=G.edge[n2][n1]['weight']
                result[x][12] +=w1*w2*w3*w4
                result[n1][12]+=w1*w2*w3*w4
                result[n2][12]+=w1*w2*w3*w4
            elif (c1):
                w3=G.edge[n1][n2]['weight']
                result[x][11] +=w1*w2*w3
                result[n1][11]+=w1*w2*w3
                result[n2][11]+=w1*w2*w3
            elif (c2):
                w3=G.edge[n2][n1]['weight']
                result[x][11] +=w1*w2*w3
                result[n1][11]+=w1*w2*w3
                result[n2][11]+=w1*w2*w3
        for n1,n2 in it.combinations(both,2):
            w1=G.edge[x][n1]['weight']
            w2=G.edge[x][n2]['weight']
            w1*=G.edge[n1][x]['weight']
            w2*=G.edge[n2][x]['weight']
            c1=G.has_edge(n1,n2)
            c2=G.has_edge(n2,n1)
            if (not c1) and (not c2):
                result[x][5] +=w1*w2
                result[n1][5]+=w1*w2
                result[n2][5]+=w1*w2
        for n1 in pre0:
            w1=G.edge[n1][x]['weight']
            for n2 in suc0:
                w2=G.edge[x][n2]['weight']
                c1=G.has_edge(n1,n2)
                c2=G.has_edge(n2,n1)
                if (c1 and c2):
                    w3=G.edge[n1][n2]['weight']
                    w4=G.edge[n2][n1]['weight']
                    result[x][10] +=w1*w2*w3*w4
                    result[n1][10]+=w1*w2*w3*w4
                    result[n2][10]+=w1*w2*w3*w4
                elif (c1):
                    w3=G.edge[n1][n2]['weight']
                    result[x][6] +=w1*w2*w3
                    result[n1][6]+=w1*w2*w3
                    result[n2][6]+=w1*w2*w3
                elif (c2):
                    w3=G.edge[n2][n1]['weight']
                    result[x][7]+=w1*w2*w3
                    result[n1][7]+=w1*w2*w3
                    result[n2][7]+=w1*w2*w3
        for n1 in pre:
            w1=G.edge[n1][x]['weight']
            for n2 in suc:
                w2=G.edge[x][n2]['weight']
                c1=G.has_edge(n1,n2)
                c2=G.has_edge(n2,n1)
                if (not c1) and (not c2):
                    result[x][2] +=w1*w2
                    result[n1][2]+=w1*w2
                    result[n2][2]+=w1*w2
        for n1 in pre0:
            w1=G.edge[n1][x]['weight']
            for n2 in both0:
                w2=G.edge[x][n2]['weight']
                w2*=G.edge[n2][x]['weight']
                c1=G.has_edge(n1,n2)
                c2=G.has_edge(n2,n1)
                if (c1 and c2):
                    w3=G.edge[n1][n2]['weight']
                    w4=G.edge[n2][n1]['weight']
                    result[x][11] +=w1*w2*w3*w4
                    result[n1][11]+=w1*w2*w3*w4
                    result[n2][11]+=w1*w2*w3*w4
                elif (c1):
                    w3=G.edge[n1][n2]['weight']
                    result[x][8] +=w1*w2*w3
                    result[n1][8]+=w1*w2*w3
                    result[n2][8]+=w1*w2*w3
                elif (c2):
                    w3=G.edge[n2][n1]['weight']
                    result[x][10] +=w1*w2*w3
                    result[n1][10]+=w1*w2*w3
                    result[n2][10]+=w1*w2*w3
        for n1 in pre:
            w1=G.edge[n1][x]['weight']
            for n2 in both:
                w2=G.edge[x][n2]['weight']
                w2*=G.edge[n2][x]['weight']
                c1=G.has_edge(n1,n2)
                c2=G.has_edge(n2,n1)
                if (not c1) and (not c2):
                    result[x][3] +=w1*w2
                    result[n1][3]+=w1*w2
                    result[n2][3]+=w1*w2
        for n1 in suc0:
            w1=G.edge[x][n1]['weight']
            for n2 in both0:
                w2=G.edge[x][n2]['weight']
                w2*=G.edge[n2][x]['weight']
                c1=G.has_edge(n1,n2)
                c2=G.has_edge(n2,n1)
                if (c1 and c2):
                    w3=G.edge[n1][n2]['weight']
                    w4=G.edge[n2][n1]['weight']
                    result[x][11] +=w1*w2*w3*w4
                    result[n1][11]+=w1*w2*w3*w4
                    result[n2][11]+=w1*w2*w3*w4
                elif (c1):
                    w3=G.edge[n1][n2]['weight']
                    result[x][10] +=w1*w2*w3
                    result[n1][10]+=w1*w2*w3
                    result[n2][10]+=w1*w2*w3
                elif (c2):
                    w3=G.edge[n2][n1]['weight']
                    result[x][9] +=w1*w2*w3
                    result[n1][9]+=w1*w2*w3
                    result[n2][9]+=w1*w2*w3
        for n1 in suc:
            w1=G.edge[x][n1]['weight']
            for n2 in both:
                w2=G.edge[x][n2]['weight']
                w2*=G.edge[n2][x]['weight']
                c1=G.has_edge(n1,n2)
                c2=G.has_edge(n2,n1)
                if (not c1) and (not c2):
                    result[x][4] +=w1*w2
                    result[n1][4]+=w1*w2
                    result[n2][4]+=w1*w2
    return result


## ordering as in milo 2004
def motifCounterFastProductSimpleGraph(G):
    result=[[0 for x in range(13)] for y in range(len(G))]
    preS=[set() for x in range(len(G))]
    for x in range(len(G)):
        for y in G[x]:
            preS[y].add(x)
    for x in range(len(G)):
#        pre=set([y for y in range(len(G)) if x in G[y]])
        pre=preS[x]
        suc=set(G[x])
        both=set(pre).intersection(suc)
        pre=pre.difference(both)
        suc=suc.difference(both)
        pre0=[y for y in pre if y<x]
        pre1=[y for y in pre if y>x]
        suc0=[y for y in suc if y<x]
        suc1=[y for y in suc if y>x]
        both0=[y for y in both if y<x]
        both1=[y for y in both if y>x]
        for n1,n2 in it.combinations(pre0,2):
            w1=G[n1][x]
            w2=G[n2][x]
            c1=n2 in G[n1]
            c2=n1 in G[n2]
            if (c1 and c2):
                w3=G[n1][n2]
                w4=G[n2][n1]
                result[x][9]+=w1*w2*w3*w4
                result[n1][9]+=w1*w2*w3*w4
                result[n2][9]+=w1*w2*w3*w4
            elif (c1):
                w3=G[n1][n2]
                result[x][6]+=w1*w2*w3
                result[n1][6]+=w1*w2*w3
                result[n2][6]+=w1*w2*w3
            elif (c2):
                w3=G[n2][n1]
                result[x][6]+=w1*w2*w3
                result[n1][6]+=w1*w2*w3
                result[n2][6]+=w1*w2*w3
        for n1,n2 in it.combinations(pre,2):
            w1=G[n1][x]
            w2=G[n2][x]
            c1=n2 in G[n1]
            c2=n1 in G[n2]
            if (not c1) and (not c2):
                result[x][1]+=w1*w2
                result[n1][1]+=w1*w2
                result[n2][1]+=w1*w2
        for n1,n2 in it.combinations(suc0,2):
            w1=G[x][n1]
            w2=G[x][n2]
            c1=n2 in G[n1]
            c2=n1 in G[n2]
            if (c1 and c2):
                w3=G[n1][n2]
                w4=G[n2][n1]
                result[x][8] +=w1*w2*w3*w4
                result[n1][8]+=w1*w2*w3*w4
                result[n2][8]+=w1*w2*w3*w4
            elif (c1):
                w3=G[n1][n2]
                result[x][6] +=w1*w2*w3
                result[n1][6]+=w1*w2*w3
                result[n2][6]+=w1*w2*w3
            elif (c2):
                w3=G[n2][n1]
                result[x][6] +=w1*w2*w3
                result[n1][6]+=w1*w2*w3
                result[n2][6]+=w1*w2*w3
        for n1,n2 in it.combinations(suc,2):
            w1=G[x][n1]
            w2=G[x][n2]
            c1=n2 in G[n1]
            c2=n1 in G[n2]
            if (not c1) and (not c2):
                result[x][0] +=w1*w2
                result[n1][0]+=w1*w2
                result[n2][0]+=w1*w2
        for n1,n2 in it.combinations(both0,2):
            w1=G[x][n1]
            w2=G[x][n2]
            w1*=G[n1][x]
            w2*=G[n2][x]
            c1=n2 in G[n1]
            c2=n1 in G[n2]
            if (c1 and c2):
                w3=G[n1][n2]
                w4=G[n2][n1]
                result[x][12] +=w1*w2*w3*w4
                result[n1][12]+=w1*w2*w3*w4
                result[n2][12]+=w1*w2*w3*w4
            elif (c1):
                w3=G[n1][n2]
                result[x][11] +=w1*w2*w3
                result[n1][11]+=w1*w2*w3
                result[n2][11]+=w1*w2*w3
            elif (c2):
                w3=G[n2][n1]
                result[x][11] +=w1*w2*w3
                result[n1][11]+=w1*w2*w3
                result[n2][11]+=w1*w2*w3
        for n1,n2 in it.combinations(both,2):
            w1= G[x][n1]
            w2= G[x][n2]
            w1*=G[n1][x]
            w2*=G[n2][x]
            c1=n2 in G[n1]
            c2=n1 in G[n2]
            if (not c1) and (not c2):
                result[x][5] +=w1*w2
                result[n1][5]+=w1*w2
                result[n2][5]+=w1*w2
        for n1 in pre0:
            w1=G[n1][x]
            for n2 in suc0:
                w2=G[x][n2]
                c1=n2 in G[n1]
                c2=n1 in G[n2]
                if (c1 and c2):
                    w3=G[n1][n2]
                    w4=G[n2][n1]
                    result[x][10] +=w1*w2*w3*w4
                    result[n1][10]+=w1*w2*w3*w4
                    result[n2][10]+=w1*w2*w3*w4
                elif (c1):
                    w3=G[n1][n2]
                    result[x][6] +=w1*w2*w3
                    result[n1][6]+=w1*w2*w3
                    result[n2][6]+=w1*w2*w3
                elif (c2):
                    w3=G[n2][n1]
                    result[x][7]+=w1*w2*w3
                    result[n1][7]+=w1*w2*w3
                    result[n2][7]+=w1*w2*w3
        for n1 in pre:
            w1=G[n1][x]
            for n2 in suc:
                w2=G[x][n2]
                c1=n2 in G[n1]
                c2=n1 in G[n2]
                if (not c1) and (not c2):
                    result[x][2] +=w1*w2
                    result[n1][2]+=w1*w2
                    result[n2][2]+=w1*w2
        for n1 in pre0:
            w1=G[n1][x]
            for n2 in both0:
                w2=G[x][n2]
                w2*=G[n2][x]
                c1=n2 in G[n1]
                c2=n1 in G[n2]
                if (c1 and c2):
                    w3=G[n1][n2]
                    w4=G[n2][n1]
                    result[x][11] +=w1*w2*w3*w4
                    result[n1][11]+=w1*w2*w3*w4
                    result[n2][11]+=w1*w2*w3*w4
                elif (c1):
                    w3=G[n1][n2]
                    result[x][8] +=w1*w2*w3
                    result[n1][8]+=w1*w2*w3
                    result[n2][8]+=w1*w2*w3
                elif (c2):
                    w3=G[n2][n1]
                    result[x][10] +=w1*w2*w3
                    result[n1][10]+=w1*w2*w3
                    result[n2][10]+=w1*w2*w3
        for n1 in pre:
            w1=G[n1][x]
            for n2 in both:
                w2=G[x][n2]
                w2*=G[n2][x]
                c1=n2 in G[n1]
                c2=n1 in G[n2]
                if (not c1) and (not c2):
                    result[x][3] +=w1*w2
                    result[n1][3]+=w1*w2
                    result[n2][3]+=w1*w2
        for n1 in suc0:
            w1=G[x][n1]
            for n2 in both0:
                w2=G[x][n2]
                w2*=G[n2][x]
                c1=n2 in G[n1]
                c2=n1 in G[n2]
                if (c1 and c2):
                    w3=G[n1][n2]
                    w4=G[n2][n1]
                    result[x][11] +=w1*w2*w3*w4
                    result[n1][11]+=w1*w2*w3*w4
                    result[n2][11]+=w1*w2*w3*w4
                elif (c1):
                    w3=G[n1][n2]
                    result[x][10] +=w1*w2*w3
                    result[n1][10]+=w1*w2*w3
                    result[n2][10]+=w1*w2*w3
                elif (c2):
                    w3=G[n2][n1]
                    result[x][9] +=w1*w2*w3
                    result[n1][9]+=w1*w2*w3
                    result[n2][9]+=w1*w2*w3
        for n1 in suc:
            w1=G[x][n1]
            for n2 in both:
                w2=G[x][n2]
                w2*=G[n2][x]
                c1=n2 in G[n1]
                c2=n1 in G[n2]
                if (not c1) and (not c2):
                    result[x][4] +=w1*w2
                    result[n1][4]+=w1*w2
                    result[n2][4]+=w1*w2
    return result
