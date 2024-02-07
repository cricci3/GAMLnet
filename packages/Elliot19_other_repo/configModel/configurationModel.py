import sys
sys.path.append('../')
sys.path.append('../../')
import configModel.simpleNetworkConfigModel as simpleNetworkConfigModel

simpleConfigModel=simpleNetworkConfigModel.directedConfigModelMk3Gen
stripNodes=simpleNetworkConfigModel.stripDeg0NodesSimpleGraph
def simpleConfigModelGen(G):
    indeg=[0 for x in range(len(G))]
    for item in G:
        for x in item:
            indeg[x]+=1
    outdeg=[len(x) for x in G]
    n=len(G)
    out1=[outdeg[x] for x in range(n)]
    in1=[indeg[x] for x in range(n)]
    weights=[G[x][y] for x in range(len(G)) for y in G[x]]
    while True:
        t1=simpleNetworkConfigModel.directedConfigModelMk3Gen(in1,out1,weights)
        t1=stripNodes(t1)
        yield t1
