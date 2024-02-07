import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
sys.path.insert(0, myPath + '/../../')
sys.path.insert(0, myPath + '/../../../')
from scipy import sparse
from scipy import stats
import netemdMethods.countOrbits.countOrbitsCython as coCy
import sys
sys.path.append('../')
import configModel.configurationModel as configModel
import networkx as nx
import random as rd
import time

import localisation.matrixTransforms as mt
#from netemdMethods import netemdFuncs as goc
from localisation import augmentation

import numpy as np
'''

## tested
def __getsparseMat__(G):
    xs=[]
    ys=[]
    ws=[]
    for x in range(len(G)):
        for y in G[x]:
            xs.append(x)
            ys.append(y)
            ws.append(float(G[x][y]))
    A=sparse.coo_matrix((ws,(xs,ys)),shape=(len(G),len(G)),dtype=np.double)
    A=sparse.csr_matrix(A)
    return A

## tested
def __getsparseMatSym__(G):
#    print 'not sure about this symmetric definition'
    A=__getsparseMat__(G)
    return A+A.T


#tested
def __fixEvector__(qw1):
    for i in range(qw1.shape[1]):
        S1=(qw1[:,i]>0).sum()
        S2=(qw1[:,i]<0).sum()
        if S1>S2:
            qw1[:,i]=qw1[:,i]
        elif S1<S2:
            qw1[:,i]=-qw1[:,i]
        else:
            ##test if sequence is symmetric
            ls1=np.array(sorted([x for x in qw1[:,i]]))
            ## normalise for numerical stability
            ls1=ls1/max(abs(ls1))
            ls2=np.array(sorted([-x for x in qw1[:,i]]))
            ls2=ls2/max(abs(ls2))
            if abs(ls1-ls2).max()<10**(-10):
                continue
            a=0
            qw2=qw1[:,i].copy()
            ## normalise for numerical stability
            qw2=qw2/max(abs(qw2))
            while abs((qw2**(2*a+1)).sum())<10**(-10):
                a+=1
#                if a==1000:
#                   import pdb
#                   pdb.set_trace()
                print('hi',a, end=' ')
            if (qw2**(2*a+1)).sum()<0:
                qw1[:,i]=-qw1[:,i]
    qw1=np.around(qw1,12)
    return qw1


## tested
def __reorder__(eigs,matrix,removeTrival=True,numEigs=5):
 #   fgt=abs(eigs[1]).sum(axis=0)
    realEval=eigs[0]
    #assert(len(fgt)==len(realEval))
    if matrix=='adj':
         #ordering=sorted(zip(-realEval,fgt,range(len(realEval))))
         ordering=sorted(zip(-realEval,list(range(len(realEval)))))
    elif matrix=='rw':
#         ordering=sorted(zip(-realEval,fgt,range(len(realEval))))
         ordering=sorted(zip(-realEval,list(range(len(realEval)))))
         if removeTrival==True:
             assert(abs(ordering[0][0]-1)<10**(14))
             ordering=ordering[1:]
    elif matrix=='adjLower':
         #ordering=sorted(zip(realEval,fgt,range(len(realEval))))
         ordering=sorted(zip(realEval,list(range(len(realEval)))))
    elif matrix=='comb':
         #ordering=sorted(zip(realEval,fgt,range(len(realEval))))
         ordering=sorted(zip(realEval,list(range(len(realEval)))))
         if removeTrival==True:
             assert(abs(ordering[0][0])<10**(14))
             ordering=ordering[1:]
    else:
         assert(False)

    ## verify there are not still ties.
    links=[]
    for i in range(len(ordering)-1):
        ## test the eigval.
        if abs(ordering[i][0]-ordering[i+1][0])<10**(-14):
           ## test the l1 norm
#           if abs(ordering[i][1]-ordering[i+1][1])<10**(-14):
              ## check they arent the same vector in distribution
              ## if they are this is fine as we are looking at localisation
              t1=np.array(sorted(eigs[1][:,ordering[i+1][-1]]))
              t2=np.array(sorted(eigs[1][:,ordering[i][-1]]))
              if (t1-t2).min()<10**(-14):
                 continue
              t2=np.array(sorted(-eigs[1][:,ordering[i][-1]]))
              if (t1-t2).min()<10**(-14):
                 continue
              ## okay let us select the order randomly
              links.append([i,i+1])

    ## okay we have ties let us randomly select
    if len(links)>0:
        Gtemp=nx.Graph()
        Gtemp.add_edges_from(links)
        for item in nx.connected_component_subgraphs(Gtemp):
              nodesInConsideration=sorted(item.nodes())
              orderNodes=[ordering[x] for x in nodesInConsideration]
              print('random',orderNodes)
              rd.shuffle(orderNodes)
              for i in range(len(nodesInConsideration)):
                    ordering[nodesInConsideration[i]]=orderNodes[i]

    ordering=[x[-1] for x in ordering]
    ordering=ordering[:numEigs]
    eigs[1]=eigs[1][:,ordering]
    eigs[0]=eigs[0][ordering]




## tested
def __getEvector__(G):
    result={}
    A=__getsparseMatSym__(G)
    if len(G)<=5:
       A1=A.todense()
       eigs=np.linalg.eigh(A1)
       eigs=list(eigs)
       eigs[1]=np.array(eigs[1])
    else:
        try:
            eigs=sparse.linalg.eigsh(A,k=5,which='LA',maxiter=100000000000)
        except:
            A1=A.todense()
            eigs=np.linalg.eigh(A1)
            eigs=list(eigs)
            eigs[1]=np.array(eigs[1])
    eigs=list(eigs)
    __reorder__(eigs,'adj')
    __fixEvector__(eigs[1])
    for i in range(eigs[1].shape[1]):
        if i<eigs[1].shape[1]:
            result[('adj',i)]=np.around(eigs[1][:,i:i+1],12)
    if len(G)<=5:
       A1=A.todense()
       eigs1=np.linalg.eigh(A1)
       eigs1=list(eigs)
       eigs1[1]=np.array(eigs[1])
    else:
        try:
            eigs1=sparse.linalg.eigsh(A,k=5,which='SA',maxiter=100000000000)
        except:
            A1=A.todense()
            eigs1=np.linalg.eigh(A1)
            eigs1=list(eigs1)
            eigs1[1]=np.array(eigs1[1])
    eigs1=list(eigs1)
    __reorder__(eigs1,'adjLower')
    __fixEvector__(eigs1[1])
    for i in range(5):
        if i<eigs1[1].shape[1]:
            result[('adjLower',i)]=np.around(eigs1[1][:,i:i+1],12)

    Lcomb=mt.buildLaplacian(A,'comb')
    eigsCombCom=mt.computeTopSpectrum(Lcomb,'comb',6)
    eigsCombSplit=[eigsCombCom['eigVals'],eigsCombCom['eigVects']]
    __reorder__(eigsCombSplit,'comb')
    eigsComb=eigsCombSplit[1]
    assert(abs(eigsComb.imag).max()<10**(-12))
    eigsComb=eigsComb.real
    __fixEvector__(eigsComb)
    for i in range(5):
        if i<eigsComb.shape[1]:
           result[('comb',i)]=np.around(eigsComb[:,i:i+1],12)

    Lrw=mt.buildLaplacian(A,'rw')
    eigsRwCom=mt.computeTopSpectrum(Lrw,'rw',6)
    eigsRwSplit=[eigsRwCom['eigVals'],eigsRwCom['eigVects']]
    __reorder__(eigsRwSplit,'rw')
    eigsRw=eigsRwSplit[1]
    #assert(abs(eigsRw.imag).max()<10**(-12))
    if eigsRw.imag.max()>10**(-12):
        print('shit imag values')
    eigsRw=eigsRw.real
#    eigsRw=abs(eigsRw)
    __fixEvector__(eigsRw)
    for i in range(5):
        if i<eigsRw.shape[1]:
            result[('rw',i)]=np.around(eigsRw[:,i:i+1],12)
    return result
#    return {'lowerAdj':eigs1[1],'upperAdj':eigs[1],'combLap':eigsComb,'rwLap':eigsRw}
#    return np.concatenate([eigs1[1],eigs[1],eigsComb,eigsRw],axis=1)


#tested
def __statsToGDD__(stats):
    result=[]
    statNames=set([y for x in stats for y in x])
    for graph in stats:
        temp1={}
        for item in graph:
            ## need to put a test here to check if the vector is constant to nearest precision
            min1=min(graph[item])
            max1=max(graph[item])
            if (max1-min1)<10**(-10):
                ## this vector is basically constant
                ### therefore can set it to any constant vector
                dfg=np.ones(graph[item].shape)
                temp1[item]=goc.getGDD(np.array(dfg))
            else:
                temp1[item]=goc.getGDD(graph[item])
        result.append(temp1)
#        relevantnulldata=[stat[item] for stat in stats if item in stat] # x[:,i:i+1] for x in stats]
#        t1=goc.__getGDDs__(relevantnulldata)
#        result[item]=t1
    return result

## tested
def __getMotifStats__(gen,numReps):
    motifStat=[]
    time0=time.time()
    for i in range(numReps):
        x=next(gen)

        ## get in and out strength stats
        inStr=[0 for node in range(len(x))]
        outStr=[0 for node in range(len(x))]
        for n1 in range(len(x)):
            for n2 in x[n1]:
                outStr[n1]+=x[n1][n2]
                inStr[n2]+=x[n1][n2]
        ## get in and out degree stats
        outStr=np.array(outStr)
        inStr=np.array(inStr)
        totStr=outStr+inStr
        t0=np.stack([outStr,inStr,totStr]).T
        t1=np.array(coCy.motifCounterFastProductSimpleGraph(x))
        t2=np.concatenate([t0,t1],axis=1)
        result={}
        for i in range(t2.shape[1]):
            result[('motif',i)]=np.around(t2[:,i:i+1],12)
        motifStat.append(result)
    time1=time.time()
    result=__statsToGDD__(motifStat)
    time2=time.time()
    print('motifStats',time1-time0)
    print('Motif GDD time',time2-time1)
    return motifStat,result

## no testing as simple wrapper on statsToGDD (tested) and getEvector (tested)
def __getEvectorStats__(gen,numReps,noRepeat=False):
    evectorStat=[]
    time0=time.time()
    for i in range(numReps):
        x=next(gen)
        count=0
        ## number required to get 5 non trival eigenvectors from the null distribution
        if noRepeat==False:
            while (len(x)<6 and count<9) or len(x)<3:
                x=next(gen)
                count+=1
        c1=__getEvector__(x)
        evectorStat.append(c1)
        del x
    time1=time.time()
    result=__statsToGDD__(evectorStat)
    time2=time.time()
    print('evectorStats',time1-time0)
    print('Evector GDD time',time2-time1)
    return evectorStat,result

## not tested as simple wrapper
def __getNetEMDstatistics__(gen,genAug,numReps,noRepeat=False):
    motifStat,motifGdd = __getMotifStats__(gen,numReps)
    evecStat,evecGdd  = __getEvectorStats__(genAug,numReps,noRepeat)
    return [motifStat,evecStat],[motifGdd,evecGdd]

class BadlySizedComparison(Exception):
    pass

## tested
def __getNetEMDdistances__(resultSet1,resultSet2):
    result={}
    motifGdds1,evecGdds1=resultSet1
    motifGdds2,evecGdds2=resultSet2
#    if len(motifGdds1)!=len(motifGdds2):
#        raise BadlySizedComparison
#    if len(evecGdds1)!=len(evecGdds2):
#        raise BadlySizedComparison
    statistics=set(y for x in [motifGdds1,motifGdds2] for y1 in x for y in y1)
    appendR=goc.robjects.r('append')
    listR=goc.robjects.r('list')
    for statistic in statistics:
        result[statistic]={}
        for i in range(len(motifGdds1)):
            if statistic in motifGdds1[i]:
                 stat1=motifGdds1[i]
                 for j in range(len(motifGdds2)):
                     if statistic in motifGdds2[j]:
                        stat2=motifGdds2[j]
                        result[statistic][(i,j)]=np.array(goc.netdist.net_emd(stat1[statistic],stat2[statistic],smoothing_window=0,method='optimise'))[0]
    statistics=set(y for x in [evecGdds1,evecGdds2] for y1 in x for y in y1 )
    for statistic in statistics:
        assert(statistic not in result)
        result[statistic]={}
        for i in range(len(evecGdds1)):
            if statistic in evecGdds1[i]:
                 stat1=evecGdds1[i]
                 for j in range(len(evecGdds2)):
                     if statistic in evecGdds2[j]:
                        stat2=evecGdds2[j]
                        result[statistic][(i,j)]=np.array(goc.netdist.net_emd(stat1[statistic],stat2[statistic],smoothing_window=0,method='optimise'))[0]
    return result


## tested
def networkxToSimpleGraph(G,returnNodes=False,mappingDict=None):
    Gnew=[{} for x in range(len(G))]
    nodes=sorted(G.nodes())
    nodesMap={nodes[i]:i for i in range(len(nodes))}
    if mappingDict==None:
        mappingDict={x:x for x in range(len(G))}
    for x in nodes:
        for y in G[x]:
            Gnew[nodesMap[x]][nodesMap[y]]=G[x][y]['weight']
    if returnNodes:
        return Gnew,nodes
    else:
        return Gnew


def netEmdAnomalyDetectionFast(G0,G0augment,nulls=None,referenceState=30,nullreps=1000,threshold=0.005,node_threshold=0.005,meth=1,nullsGraphsL=None,refGraphsL=None):

    ## Save node order
    nodes=sorted(G0.nodes())
    nodesAug=sorted(G0augment.nodes())
    assert(nodes==nodesAug)
    if len(nodes)<4:
        ## motifs cannot find a structure here
        ## eigenvectors might but unlikely, better just to have a small community category
        return [['smallCommunityCatNetEMD',nodes,[1,]*len(nodes)],],[]


    ## Remove labels from graph
    G=nx.convert_node_labels_to_integers(G0,ordering='sorted')
    Gaugment=nx.convert_node_labels_to_integers(G0augment,ordering='sorted')

    ## Convert graph to simple grap:
    G1=networkxToSimpleGraph(G)
    G1augment=networkxToSimpleGraph(Gaugment)

    ## gen models
    gen=configModel.simpleConfigModelGen(G1)
    genAug=configModel.simpleConfigModelGen(G1augment)

    ## Get stats
    refStats,refGdds     = __getNetEMDstatistics__(gen,genAug,referenceState)
    nullStats,nullGdds   = __getNetEMDstatistics__(gen,genAug,nullreps)

    ## Computing the null statistics'
    nullDistances=__getNetEMDdistances__(refGdds,nullGdds)
    print('computed the null distances')

    ## Computing trimmed means
    nulls={}

    for statistic in nullDistances:

        curNull=nullDistances[statistic]

        ## extract all of the null distances
        nulls[statistic]=[]
        for i in range(nullreps):

            currentNull=[]
            ## reference graphs are the first set of graphs
            for j in range(referenceState):
                if (j,i) in curNull:
                   currentNull.append(curNull[(j,i)])
            if len(currentNull)>0:
               t1=stats.trim_mean(currentNull,0.1,axis=0)
               nulls[statistic].append(t1)

    ## Real stats
    realStats,realGdds=__getNetEMDstatistics__(iter([G1,]),iter([G1augment,]),1,noRepeat=True)

    ## Get real distances
    realDistances=__getNetEMDdistances__(refGdds,realGdds)

    features=[]
    result=[]
    resultStructs=[]

    for statistic in realDistances:
        statDistances=realDistances[statistic]
        assert(max([0,]+[x[1] for x in statDistances])==0)
        currentDist=[]

        ## reference state is passed first a
        ## and there is only one test graph
        ## make currentDist between the reference set the test graph
        for j in range(referenceState):
            if (j,0) in statDistances:
               currentDist.append(statDistances[(j,0)])

        ## compute the trim mean
        t2=stats.trim_mean(currentDist,0.1,axis=0)
#        print statistic,t2,np.array(nulls[statistic]).mean(),np.array(nulls[statistic]).std()
        ## compute the numerator
        num=1+sum(y>=t2 for y in nulls[statistic])
        ## compute the denominator
        denom=float(len(nulls[statistic]))+1.0

        pval=num/denom

        if pval<threshold:
            ## need to compute the nodes that deviate.
            if statistic in realStats[0][0]:
               assert(statistic not in realStats[1][0])
               r0=realStats[0][0][statistic]
            else:
               r0=realStats[1][0][statistic]

            r1=r0-r0.mean()
            if r1.std()!=0:
                r1/=r1.std()
            r1=abs(r1)
            r1[r1<2]=0

            result.append([statistic,'cutout2',nodes,r1])

            r1=r0-r0.mean()
            if r1.std()!=0:
                r1/=r1.std()
            r1=abs(r1)
            thres=np.percentile(r1,95)
            r1[r1<thres]=0

            weight=-stats.norm.ppf(pval)
            r1[r1>=thres]=weight
            result.append([statistic,'thres5',nodes,r1])


#            node2score=dict(list(zip(nodes,r1)))
#            nodetempList=[]
#            for i123 in range(len(nodes)):
#                if r1[i123]>0:
#                    nodetempList.append(nodes[i123])
#            Gtemp=G0.subgraph(nodetempList)
#            for Gconnect in nx.weakly_connected_component_subgraphs(Gtemp):
#                t1=[[n12,node2score[n12]] for n12 in Gconnect]
#                resultStructs.append([Gconnect,t1])
    return result,resultStructs

'''