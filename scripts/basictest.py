#Source:
#  A. Elliott, M. Cucuringu, M. M. Luaces, P. Reidy, and G. Reinert, 
#   “Anomaly detection in networks with application to financial transaction networks,” 2019. 
#   [Online]. Available: https://arxiv.org/abs/1901.00402

import numpy as np
from scipy import stats
from collections import Counter
import random as rd
#from configModel import configurationModel
import networkx as nx
from math import ceil
try:
    import community
except:
    print('Community detection package not found')
def __makeUndirected__(G):
    G1=nx.Graph()
    for x in G:
        for y in G[x]:
            w=G[x][y]['weight']
            if x in G[y]:
                w+=G[y][x]['weight']
            G1.add_edge(x,y,weight=w)
    return G1


def clusterLouvain(G,save=False,path='',saveGraph=None):
    if saveGraph==None:
        saveGraph=G
    print('starting Cluster')
    G1=__makeUndirected__(G)
    coms=community.best_partition(G1)
    print('end Cluster')
    res=[]
    for x in range(max(coms.values())+1):
        nodes=[y for y in coms if coms[y]==x]
        res.append(nodes)
    return res

#tested
def __getDensity__(G,com):
    assert(len(com)>1)
    l1=len(com)
    com=set(com)
    assert(l1==len(com))
    count=0
    for x in com:
        for y in G[x]:
            if y in com:
                count+=1
    t0=count/float(len(com)*(len(com)-1))
    return t0

def __getPval__(t1,nulls):
    numerator=1+sum(x>=t1 for x in nulls)
    denominator=float(len(nulls)+1)
    pval=numerator/denominator
    return pval


def subNetworkDensityTest(G,coms,reps):
    G0=graphUtils.networkxToSimpleGraph(G)
    gen1=configurationModel.simpleConfigModelGen(G0)
    gen2=(graphUtils.simpleGraphToNetworkx(next(gen1)) for x in range(10000))
    nulls=[]
    com=coms[0]
    result={}
    for rep in range(reps):
        Gnull=next(gen2)
        Gnull1=graphUtils.modNetworkRep(Gnull.copy(),inPlace=False)
        coms1=clusterLouvain(Gnull1)
        com1=rd.choice(coms1)
        t1=__getDensity__(Gnull,com1)
        nulls.append(t1)
    results=[]
    for com in coms:
        t1=__getDensity__(G,com)
        pval=__getPval__(t1,nulls)
        for x in com:
            if stats.norm.ppf(1-pval,0,1)>0:
                 result[x]=stats.norm.ppf(1-pval,0,1)
    return [['communityDensity','configModelTest',result],]


#tested
def strengthDegree(G,numReps,pvalThres):
    assert(nx.DiGraph==type(G))
    weights=[G[x][y]['weight'] for x in G for y in G[x]]
    weights=np.array(weights)
    dfg=Counter(weights)
    strengths={x:[] for x in G}
    for x in G:
        for y in G[x]:
            strengths[x].append(G[x][y]['weight'])
            strengths[y].append(G[x][y]['weight'])
    result={x:0 for x in G}
    nulls={}
    count=0
    for k in set([len(x) for x in list(strengths.values())]):
        count+=1
        temp1=np.random.choice(weights,size=[k,numReps],replace=True)
        nulls[k]=stats.gmean(temp1)
        nulls[k]=list(nulls[k])
    for x in G.nodes():
        ## compute the
        k=len(strengths[x])
        s1=stats.gmean(strengths[x])
        temp1=0
        numerator=1
        denom=len(nulls[k])+1.0
        for i in range(len(nulls[k])):
             if nulls[k][i]>=s1 or abs(s1-nulls[k][i])<10**(-14):
                  numerator+=1
        pval=numerator/denom
        if pval<=pvalThres:
            result[x]=stats.norm.ppf(1-pval,0,1)
    return [['basicStats','strength',result],]


#tested
def strengthDegree10(G,numReps,pvalThres):
    weights=[G[x][y]['weight'] for x in G for y in G[x]]
    strengths={x:[] for x in range(len(G))}
    totalDegree={x:0 for x in range(len(G))}
    for x in range(len(G)):
        totalDegree[x]+=len(G[x])
        for y in G[x]:
            totalDegree[y]+=1
            strengths[x].append(G[x][y]['weight'])
            strengths[y].append(G[x][y]['weight'])
    for x in strengths:
        k=len(strengths[x])
        num=int(ceil(0.1*k))
        temp1=sorted(strengths[x])[-num:]
        strengths[x]=stats.gmean(temp1)
    nulls={}
    count=0
    for k in set(totalDegree.values()):
        count+=1
        nulls[k]=[]
        temp1=np.random.choice(weights,size=[k,numReps])
        temp1.sort(axis=0)
        num=int(ceil(0.1*k))
        temp2=temp1[-num:,:]
        temp3=stats.gmean(temp2)
        temp3.sort()
        nulls[k]=list(temp3)
    result={x:0 for x in G}
    for x in G:
        ## compute the
        k=totalDegree[x]
        s1=strengths[x]
        temp1=0
        numerator=1
        numerator=1
        denom=len(nulls[k])+1.0
        for i in range(len(nulls[k])):
             if nulls[k][i]>=s1 or abs(s1-nulls[k][i])<10**(-14):
                  numerator+=1
        pval=numerator/denom
        if pval<=pvalThres:
            result[x]=stats.norm.ppf(1-pval,0,1)
    return [['basicStats','strength10',result],]



#tested
def strengthDegree20(G,numReps,pvalThres):
    weights=[G[x][y]['weight'] for x in G for y in G[x]]
    strengths={x:[] for x in G}
    totalDegree={x:0 for x in G}
    for x in G:
        totalDegree[x]+=len(G[x])
        for y in G[x]:
            totalDegree[y]+=1
            strengths[x].append(G[x][y]['weight'])
            strengths[y].append(G[x][y]['weight'])
    for x in strengths:
        k=len(strengths[x])
        num=int(ceil(0.2*k))
        temp1=sorted(strengths[x])[-num:]
        strengths[x]=stats.gmean(temp1)
    result={x:0 for x in G}
    nulls={}
    count=0
    for k in set(totalDegree.values()):
        count+=1
        nulls[k]=[]
        temp1=np.random.choice(weights,size=[k,numReps])
        temp1.sort(axis=0)
        num=int(ceil(0.2*k))
        temp2=temp1[-num:,:]
        temp3=stats.gmean(temp2)
        temp3.sort()
        nulls[k]=list(temp3)
    for x in range(len(G)):
        ## compute the
        k=totalDegree[x]
        s1=strengths[x]
        temp1=0
        numerator=1
        denom=len(nulls[k])+1.0
        for i in range(len(nulls[k])):
             if nulls[k][i]>=s1 or abs(s1-nulls[k][i])<10**(-14):
                  numerator+=1
        pval=numerator/denom
        if pval<=pvalThres:
            result[x]=stats.norm.ppf(1-pval,0,1)
    return [['basicStats','strength20',result],]


#tested
def strengthCom(G,coms):
    weights=[G[x][y]['weight'] for x in range(len(G)) for y in G[x]]
    weights=np.array(weights)
    fullStat=stats.mstats.gmean(weights)
    results={x:0 for x in G}
    results1={x:0 for x in G}
    for com in coms:
        com_s=set(com)
        weightsCom=[G[x][y]['weight'] for x in com for y in G[x] if y in com_s]
        weightsCom=np.array(weightsCom)
        comStat=stats.mstats.gmean(weightsCom)
        stat=comStat/fullStat
        stat1=stat/float(len(com))
        for x in com:
            results[x]=stat
            results1[x]=stat1
    fullRes=[]
    fullRes.append(['strengthCom','full',results])
    fullRes.append(['strengthCom','aver',results1])
    return fullRes

#tested
def densityCom(G,coms):
    results={x:0 for x in G}
    results1={x:0 for x in G}
    d1=sum(len(G[x]) for x in G)/float(len(G)*(len(G)-1))
    for com in coms:
        com_s=set(com)
        dCom=sum([1 for x in com for y in G[x] if y in com_s])/float(len(com)*(len(com)-1))
        stat=dCom/float(d1)
        stat1=stat/float(len(com))
        for x in com:
            results[x]=stat
            results1[x]=stat1
    fullRes=[]
    fullRes.append(['densityCom','full',results])
    fullRes.append(['densityCom','aver',results1])
    return fullRes

def getTotalDegree(G):
    indeg=G.in_degree()
    outdeg=G.out_degree()
    result={x:indeg[x]+outdeg[x] for x in G}
    mean=np.mean(list(result.values()))
    std=np.std(list(result.values()))
    for x in result:
        result[x]=(result[x]-mean)/std
    fullRes=[]
    fullRes.append(['standardiseDegree',result])
    return fullRes

