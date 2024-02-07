import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
sys.path.insert(0, myPath + '/../../')
sys.path.insert(0, myPath + '/../../../')

from math import floor
from math import sqrt
import configModel.configurationModel as configM
import localisation.matrixTransforms as mt
from localisation import augmentation
import networkx as nx
from scipy import stats
from scipy import sparse
import numpy as np
import random as rd



## Custom exception
class NonSensicalSplit(Exception):
    pass



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
def __getEvector__(G,numEigs,returnEvals=False,direction='LA'):
    A=__getsparseMatSym__(G)
    if numEigs==len(G):
        A=A.todense()
        eigs=np.linalg.eigh(A)
        eigs=list(eigs)
        eigs[1]=np.array(eigs[1])
    else:

        try:
            eigs=sparse.linalg.eigsh(A,k=numEigs,which=direction,maxiter=100000000000)
        except:
            A=A.todense()
            eigs=np.linalg.eigh(A)
            eigs=list(eigs)
            eigs[1]=np.array(eigs[1])
    ## breaks ties with sum of absolute values.
    if direction=='LA':
#        fgt=abs(eigs[1]).sum(axis=0)
        #ordering=sorted(zip(-eigs[0],fgt,range(len(eigs[1]))))
        ordering=sorted(zip(-eigs[0],list(range(len(eigs[1])))))
    elif direction=='SA':
#        fgt=abs(eigs[1]).sum(axis=0)
#        ordering=sorted(zip(eigs[0],fgt,range(len(eigs[1]))))
        ordering=sorted(zip(eigs[0],list(range(len(eigs[1])))))
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


    ## Get ordering
    ordering=ordering[:numEigs]
    ordering=[x[1] for x in ordering]

    ## reorder based on this ordering
    if returnEvals:
        return eigs[0][ordering],eigs[1][:,ordering]
    else:
        result=eigs[1][:,ordering]
        return result
   # return eigs[1]

#tested
def __getStat__(G1,mat,numEigs,returnEvals=False,removeTrival=True):
    if mat=='adj':
        if returnEvals:
            realEval,realEvec=__getEvector__(G1,numEigs,returnEvals)
        else:
            realEvec=__getEvector__(G1,numEigs,returnEvals)
    elif mat=='adjLower':
        if returnEvals:
            realEval,realEvec=__getEvector__(G1,numEigs,returnEvals,'SA')
        else:
            realEvec=__getEvector__(G1,numEigs,returnEvals,'SA')
    elif mat in ['comb','rw']:
        A=__getsparseMatSym__(G1)
        assert(A.sum(axis=0).min()!=0)
        assert(A.sum(axis=1).min()!=0)
        L=mt.buildLaplacian(A,mat)
        if removeTrival==True:
           # can only add an extra eigenvalue if there is one
           real=mt.computeTopSpectrum(L,mat,min(numEigs+1,len(G1)))
        else:
           real=mt.computeTopSpectrum(L,mat,numEigs)
        realEval=real['eigVals']
#        fgt=abs(real['eigVects']).sum(axis=0)
#        assert(len(fgt)==len(realEval))
        if mat=='comb':
            #ordering=sorted(zip(realEval,fgt,range(len(realEval))))
            ordering=sorted(zip(realEval,list(range(len(realEval)))))
            if removeTrival==True:
                 assert(abs(ordering[0][0])<10**(14))
                 ordering=ordering[1:]
        elif mat=='rw':
            #ordering=sorted(zip(-realEval,fgt,range(len(realEval))))
            ordering=sorted(zip(-realEval,list(range(len(realEval)))))
            if removeTrival==True:
                 assert(abs(ordering[0][0]-1.0)<10**(14))
                 ordering=ordering[1:]

        ## verify there are not still ties.
        links=[]
        for i in range(len(ordering)-1):
            if abs(ordering[i][0]-ordering[i+1][0])<10**(-14):
             #  if abs(ordering[i][1]-ordering[i+1][1])<10**(-14):
                  ## check they arent the same vector in distribution
                  ## they are this is fine as we are looking at localisation
                  t1=np.array(sorted(real['eigVects'][:,ordering[i+1][-1]]))
                  t2=np.array(sorted(real['eigVects'][:,ordering[i][-1]]))
                  if (t1-t2).min()<10**(-14):
                     continue
                  t2=np.array(sorted(-real['eigVects'][:,ordering[i][-1]]))
                  if (t1-t2).min()<10**(-14):
                     continue
                  links.append([i,i+1])

        ## okay we have ties let us randomly select
        if len(links)>0:
            Gtemp=nx.Graph()
            Gtemp.add_edges_from(links)
            for item in nx.connected_component_subgraphs(Gtemp):
                  nodesInConsideration=sorted(item.nodes())
                  orderNodes=[ordering[x] for x in nodesInConsideration]
                  rd.shuffle(orderNodes)
                  for i in range(len(nodesInConsideration)):
                        ordering[nodesInConsideration[i]]=orderNodes[i]

        ordering=ordering[:numEigs]
        ordering=[x[1] for x in ordering]
        realEval=realEval[ordering]
        realEvec=real['eigVects'][:,ordering]
    else:
        print('mat not defined')
        assert(False)
    if returnEvals:
        return realEval,realEvec
    else:
        return realEvec

#tested
def __computeL4norm__(x):
    return (x**4).sum(axis=0)

#tested
def __computeMinPosNeg__(x):
    t1=(x>10**-16).sum(axis=0)
    t2=((-x)>10**-16).sum(axis=0)
    t1[t1==0]=x.shape[1]
    t2[t2==0]=x.shape[1]
    return np.minimum(t1,t2)/float(x.shape[1])

#tested
def __getExpNorm__(x):
    return (np.exp(abs(x))-abs(x)-1).sum(axis=0)






## tested
def __averageLargestNodeNull__(eVec,pval,nullEvec):
    maxNull=[max(abs(x)) for x in nullEvec]
    maxNull.sort(key=lambda x: -x)
    maxNull=np.array(maxNull)
    averageMaxNull=maxNull.mean()

    ## pseudo p-values of each node
    result1=[]
    for node in range(eVec.shape[0]):
        numerator=1.0
        for j in range(len(maxNull)):
            if maxNull[j]>=abs(eVec[node]):
                numerator+=1.0
        nodePval=numerator/float(len(maxNull)+1)
        ## this shouldnt ever happen but just in case.
        if nodePval<0.5:
            weight=-stats.norm.ppf(nodePval)
            result1.append(weight)
        else:
            result1.append(0)

    ## raw values after the threshold
    result2=abs(eVec)
    result2[result2<averageMaxNull]=0

    ## threshold with weight prob of eigenvector
    if pval<0.5:
        weight=-stats.norm.ppf(pval)
    else:
        ## this will only happen if the pvalue threshold is set to above 0.5
        print('pval above 0.5 passed to sign method set to zero')
        weight=0
    result3=(result2>0).astype(np.double)
    result3*=weight
    result4=abs(eVec)*(sqrt(len(eVec)))
    return result1,result2,result3,result4


## wrapper around the other routine
def __getL4normNode__(realEvec,i,pval,nulls,nodes):
    eVec=realEvec[:,i]**4
    nullEvec=[x[:,i]**4 for x in nulls if x.shape[1]>i]
    t1,t2,t3,t4=__averageLargestNodeNull__(eVec,pval,nullEvec)
    result=[]
    result.append(['nodeStat','l4AverLargeNodePval',i,nodes,t1])
    result.append(['nodeStat','l4AverLargeNodeThres',i,nodes,t2])
    result.append(['nodeStat','l4AverLargeNodeThresPval',i,nodes,t3])
    result.append(['nodeStat','l4AverLargeNodeNoThres',i,nodes,t4])
    return result


## trival no needs for a test
def __getL4normStruct__(realEvec,i,pval,nulls,G,nodes):
    return ['structStat','l4norm','no result can be computed from the node stat']
## trival no needs for a test
def __getExpNormStruct__(realEvec,i,pval,nulls,G,nodes):
    return ['structStat','Expnorm','no result can be computed from the node stat']

## wrapper around the other routine
def __getExpNormNode__(realEvec,i,pval,nulls,nodes):
    eVec=np.exp(abs(realEvec[:,i]))-abs(realEvec[:,i])-1
    nullEvec=[np.exp(abs(x[:,i]))-abs(x[:,i])-1 for x in nulls if x.shape[1]>i]
    t1,t2,t3,t4=__averageLargestNodeNull__(eVec,pval,nullEvec)
    result=[]
    result.append(['nodeStat','expAverLargeNodePval',i,nodes,t1])
    result.append(['nodeStat','expAverLargeNodeThres',i,nodes,t2])
    result.append(['nodeStat','expAverLargeNodeThresPval',i,nodes,t3])
    result.append(['nodeStat','expAverLargeNodeNoThres',i,nodes,t4])
    return result

#tested
def __per90contribL4__(x):
    result=[]
    for i in range(x.shape[1]):
        temp1=sorted(abs(x[:,i]**4))
        sum1=sum(temp1)
        curAmount=0
        count=len(temp1)-1
        count1=0
        while curAmount<0.9*sum1:
            t1=temp1[count]
            while temp1[count]==t1 and count!=-1:
                curAmount+=temp1[count]
                count-=1
                count1+=1
        result.append(count1/float(len(temp1)))
    return np.array(result)


#tested
def __per90contribNodeL4__(realEvec,i,pval,nulls,nodes):
    result=[]
    vals=abs(realEvec[:,i]**4)
    sum1=sum(vals)
    temp1=sorted(zip(abs(realEvec[:,i]**4),list(range(realEvec.shape[0]))))
    curAmount=0
    count=len(temp1)-1
    count1=0
    result=[]
    while curAmount<0.9*sum1:
        t1=temp1[count][0]
        while temp1[count][0]==t1 and count!=-1:
            curAmount+=temp1[count][0]
            result.append(temp1[count][1])
            count-=1
            count1+=1
    weight=-stats.norm.ppf(pval)
    resultVec=np.zeros(realEvec.shape[0])
    for x in result:
        resultVec[x]=weight
    return [['nodeStat','per90ContribL4',i,nodes,resultVec],]

#tested
def __per90contribStructL4__(realEvec,i,pval,nulls,G,nodes):
    result=[]
    vals=abs(realEvec[:,i]**4)
    sum1=sum(vals)
    temp1=sorted(zip(abs(realEvec[:,i]**4),list(range(realEvec.shape[0]))))
    curAmount=0
    count=len(temp1)-1
    count1=0
    result=[]
    weight=-stats.norm.ppf(pval)
    while curAmount<0.9*sum1:
        t1=temp1[count][0]
        while temp1[count][0]==t1 and count!=-1:
            curAmount+=temp1[count][0]
            result.append(temp1[count][1])
            count-=1
            count1+=1
    n1=[nodes[x] for x in result]
    G1=G.subgraph(n1)
    return [['structStat','per90ContribL4',i,weight,G.subgraph(n1)],]


#tested
def __per90contrib__(x):
    result=[]
    for i in range(x.shape[1]):
        temp1=sorted(abs(x[:,i]))
        sum1=sum(temp1)
        curAmount=0
        count=len(temp1)-1
        count1=0
        while curAmount<0.9*sum1:
            t1=temp1[count]
            while temp1[count]==t1 and count!=-1:
                curAmount+=temp1[count]
                count-=1
                count1+=1
        result.append(count1/float(len(temp1)))
    return np.array(result)


#tested
def __per90contribNode__(realEvec,i,pval,nulls,nodes):
    result=[]
    vals=abs(realEvec[:,i])
    sum1=sum(vals)
    temp1=sorted(zip(abs(realEvec[:,i]),list(range(realEvec.shape[0]))))
    curAmount=0
    count=len(temp1)-1
    count1=0
    result=[]
    while curAmount<0.9*sum1:
        t1=temp1[count][0]
        while temp1[count][0]==t1 and count!=-1:
            curAmount+=temp1[count][0]
            result.append(temp1[count][1])
            count-=1
            count1+=1
    weight=-stats.norm.ppf(pval)
    resultVec=np.zeros(realEvec.shape[0])
    for x in result:
        resultVec[x]=weight
    return [['nodeStat','per90Contrib',i,nodes,resultVec],]

#tested
def __per90contribStruct__(realEvec,i,pval,nulls,G,nodes):
    result=[]
    vals=abs(realEvec[:,i])
    sum1=sum(vals)
    temp1=sorted(zip(abs(realEvec[:,i]),list(range(realEvec.shape[0]))))
    curAmount=0
    count=len(temp1)-1
    count1=0
    result=[]
    weight=-stats.norm.ppf(pval)
    while curAmount<0.9*sum1:
        t1=temp1[count][0]
        while temp1[count][0]==t1 and count!=-1:
            curAmount+=temp1[count][0]
            result.append(temp1[count][1])
            count-=1
            count1+=1
    n1=[nodes[x] for x in result]
    G1=G.subgraph(n1)
    return [['structStat','per90Contrib',i,weight,G.subgraph(n1)],]

#tested
def __computeMinPosNegNode__(realEvec,i,pval,nulls,nodes):
    gtZero=(realEvec[:,i]>10**-16)
    ltZero=((-realEvec[:,i])>10**-16)
    t1=(gtZero).sum(axis=0)
    t2=(ltZero).sum(axis=0)
    if t1==0 and t2==0:
        raise NonSensicalSplit
    if t1==0:
        t1=realEvec.shape[0]
    if t2==0:
        t2=realEvec.shape[0]
    weight=-stats.norm.ppf(pval)
#    if t1==t2 or min(t1,t2)>=floor(realEvec.shape[0]/2.0):
    if min(t1,t2)>=floor(realEvec.shape[0]/2.0):
        ## If this happens then there is no localisation, but the nulls all
        #have one sign
        te1=(gtZero).sum(axis=0)
        te2=(ltZero).sum(axis=0)
        if te1==realEvec.shape[0] or te2==realEvec.shape[1]:
            result=[['nodeStat','sideStatLargeNum',i,nodes,(gtZero+ltZero)*weight],]
            result.append(['nodeStat','sideStatAverageLargeNum',i,nodes,(gtZero+ltZero)*weight/float(ltZero.sum()+gtZero.sum())])
            return result
        if t1==t2:
            result=[['nodeStat','sideStatEqualLargeNum',i,nodes,(gtZero+ltZero)*weight],]
            result.append(['nodeStat','sideStatAverageEqualLargeNum',i,nodes,(gtZero+ltZero)*weight/float(ltZero.sum()+gtZero.sum())])
            return result

        if t1<t2:
            result=[['nodeStat','sideStatLargeNum',i,nodes,gtZero*weight],]
            result.append(['nodeStat','sideStatAverageLargeNum',i,nodes,gtZero*weight/float(gtZero.sum())])
            return result
        else:
            result=[['nodeStat','sideStatLargeNum',i,nodes,ltZero*weight],]
            result.append(['nodeStat','sideStatAverageLargeNum',i,nodes,ltZero*weight/float(ltZero.sum())])
            return result

    if t1==t2:
        result=[['nodeStat','sideStatEqual',i,nodes,(gtZero+ltZero)*weight],]
        result.append(['nodeStat','sideStatAverageEqual',i,nodes,(gtZero+ltZero)*weight/float(ltZero.sum()+gtZero.sum())])
        return result

    if t1<t2:
        result=[['nodeStat','sideStat',i,nodes,gtZero*weight],]
        result.append(['nodeStat','sideStatAverage',i,nodes,gtZero*weight/float(gtZero.sum())])
        return result
    else:
        result=[['nodeStat','sideStat',i,nodes,ltZero*weight],]
        result.append(['nodeStat','sideStatAverage',i,nodes,ltZero*weight/float(ltZero.sum())])
        return result
    assert(False)

#tested
def __computeMinPosNegStruct__(realEvec,i,pval,nulls,G,nodes):
    return 'not defined for now'
    gtZero=(realEvec[:,i]>10**-16)
    ltZero=((-realEvec[:,i])>10**-16)
    t1=(gtZero).sum(axis=0)
    if t1==0:
        t1=realEvec.shape[0]
    t2=(ltZero).sum(axis=0)
    if t2==0:
        t2=realEvec.shape[0]
    weight=-stats.norm.ppf(pval)
    if t1==t2 or min(t1,t2)>=floor(realEvec.shape[0]/2.0):
        ## If this happens then there is no localisation, but the nulls all
        #have one sign
        raise NonSensicalSplit
    elif t1<t2:
        n1=[nodes[j] for j in range(len(nodes)) if gtZero[j]]
        return [['structStat','sideStat',i,weight,G.subgraph(n1)],]
    else:
        n1=[nodes[j] for j in range(len(nodes)) if ltZero[j]]
        return [['structStat','sideStat',i,weight,G.subgraph(n1)],]
    assert(False)


#tested
def __sigFinder__(methNum, realLocStats, nullLocStats, realEvec, methNodes, methStructs, nulls,pvalThres,tail,G,nodes):
    resultNode=[]
    resultStruct=[]
    nullStats=[x[methNum] for x in nullLocStats]
    realLocStat=realLocStats[methNum]
    methNode=methNodes[methNum]
    methStruct=methStructs[methNum]
    #assert(len(set(len(x) for x in nullStats))==1)
    for i in range(len(realLocStat)):
        local_null=[]
	## restrict to the number of local_nulls which have enough eigenvalues
        for x in nullStats:
             if len(x)>i:
                 local_null.append(x[i])
        #print local_null
#        local_null=[x[i] for x in nullStats]
        real=realLocStat[i]
        if tail=='upper':
            numerator=1+sum(real<=x for x in local_null)
        elif tail=='lower':
            numerator=1+sum(real>=x for x in local_null)
        else:
            assert(False)
        pval=numerator/(float(len(local_null))+1)
        if (pval<=pvalThres):
            resultNode   += methNode(realEvec,i,pval,nulls,nodes)
            resultStruct += methStruct(realEvec,i,pval,nulls,G,nodes)
    return resultNode,resultStruct


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

def localisationWithRandomisationSimpleGraph(G,mat='adj',numEigs=100,nullreps=1000,pvalThres=0.005,reps=None):
    ## Convert to simple graph
    G1,nodes=networkxToSimpleGraph(G,True)

    if len(nodes)<4:
        ## motifs cannot find a structure here
        ## eigenvectors might but unlikely, better just to have a small community category
        return [['smallCommunityCatLocal',nodes,[1,]*len(nodes)],],[],[]
    ## Get a configuration model generator
    generator=configM.simpleConfigModelGen(G1)

    ## Restrict the number of eigs to the size of the matrix
    if numEigs>=len(G1):
        if mat=='adj' or mat=='adjLower':
              numEigs=len(G1)
        elif mat=='comb':
              numEigs=len(G1)-1
        elif mat=='rw':
              numEigs=len(G1)-1
        else:
              assert(False)


    ## Get Real eVec
    print('get Real Evec')
    realEvec=__getStat__(G1,mat,numEigs)

    ## add raw eigenvalues
    resultNode=[]
    for i in range(numEigs):
        resultNode.append(['nodeStat','rawAbsEvec',i,abs(realEvec[:,i])])
    ## List of localisation statistics
    methods          = [__computeL4norm__, __getExpNorm__, __computeMinPosNeg__,__per90contrib__,__per90contribL4__]
    methodGetNodes   = [__getL4normNode__, __getExpNormNode__, __computeMinPosNegNode__,__per90contribNode__,__per90contribNodeL4__]
    methodGetStructs = [__getL4normStruct__, __getExpNormStruct__, __computeMinPosNegStruct__, __per90contribStruct__,__per90contribStructL4__]
    methodNames      = ['L4 Norm','Exp Norm','minPosNeg','Per90Contrib','Per90ContribL4']
    methodDirections = ['upper','upper','lower','lower','lower']
    ## get real stats
    print('get Real Stats')
    realLocStats=[x(realEvec) for x in methods]

    nulls=[]
    nullLocStats=[]


    ## get null replicates
    print('nullreps:', end=' ')
    nullNetworkSizes=[]
    for x in range(nullreps):

        ## Get configuration graph
        Gn1=next(generator)

        ## resample if too small
        count=1
        while len(Gn1)<numEigs:
            Gn1=next(generator)
            count+=1
            if count==10 and len(Gn1)>2:
               break
        nullNetworkSizes.append(len(Gn1))
        ## Get eVec
        ## Get numEigs the dimension is big enough
        ## Otherise get the size of the graph
        if len(Gn1)<numEigs:
           eVec=__getStat__(Gn1,mat,len(Gn1))
        else:
           eVec=__getStat__(Gn1,mat,numEigs)

        ## Compute null localisation statistics
        nullLocStat=[x(eVec) for x in methods]
        nullLocStats.append(nullLocStat)

        ## store the nulls
        nulls.append(eVec)
        if len(nullNetworkSizes)%10==0:
            print(len(nullNetworkSizes), end=' ')
            import sys
            sys.stdout.flush()

    #print 'nulls generated'
    resultNode=[]
    resultStruct=[]

    for methNum in range(len(methods)):
        methodName=methodNames[methNum]
        print(methodName, end=' ')
        sys.stdout.flush()
        tail=methodDirections[methNum]
#        resultN,resultS = __sigFinder__(methodNum,nullStats,realLocStat,methodGetNodes,methodGetStructs)
        resultN,resultS=__sigFinder__(methNum, realLocStats, nullLocStats,
                realEvec, methodGetNodes, methodGetStructs,
                nulls,pvalThres,tail,G,nodes)
        resultNode+=resultN
        resultStruct+=resultS
    return resultNode,resultStruct,nullNetworkSizes

def localisationAllMatsWrapper(G,nullreps=1000):
    results={}
    localStats = []
    localStats.append('expAverLargeNodeNoThres')
    localStats.append('expAverLargeNodePval')
    localStats.append('expAverLargeNodeThres')
    localStats.append('expAverLargeNodeThresPval')
    localStats.append('l4AverLargeNodeNoThres')
    localStats.append('l4AverLargeNodePval')
    localStats.append('l4AverLargeNodeThres')
    localStats.append('l4AverLargeNodeThresPval')
    localStats.append('per90Contrib')
    localStats.append('per90ContribL4')
    localStats.append('sideStat')
    localStats.append('sideStatAverage')
    localStats.append('sideStatAverageEqual')
    localStats.append('sideStatEqual')
    mats = ['adj','comb','rw','adjLower']
    for mat in mats:
        for l1 in localStats:
            results[str(mat)+'_'+str(l1)] = {x:0 for x in G}
    G1=augmentation.modNetworkRep(G)
    for mat in ['adj','adjLower','comb','rw']:
        res,res1,res2 = localisationWithRandomisationSimpleGraph(G1,mat,20,nullreps,0.05)
        for item in res:
    #        if (mat, item[1]) not in results:
    #            results[str(mat)+'_'+str(item[1])]={x:0 for x in G}
            for i in range(len(item[3])):
                results[str(mat)+'_'+str(item[1])][item[3][i]]+=item[4][i]
    return results
