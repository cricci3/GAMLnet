from collections import Counter
from math import floor
import sys
from configModel import configurationModel as configM
from pathFinder import pathFinderCython
probablisticMethodCyImport=True
from scipy import stats



def pathFinderRandomComparision(G1,beamwidth,maxLen,reps=100,pvalThreshold=0.05):
    nulls=[]
#    G1=graphUtils.networkxToSimpleGraph(G)
    gen=configM.simpleConfigModelGen(G1)
    nulls=(next(gen) for x in range(reps))
    return __pathFinderRandomComparisionHelper__(G1,nulls,beamwidth,maxLen,pvalThreshold=0.05)

##tested
def __pathFinderCythonChoice__(G,beamwidth,maxLen):
    return pathFinderCython.__pathFinderHelper__(G,beamwidth,maxLen)

##tested
def __pathFinderRandomComparisionHelper__(G,nulls,beamwidth,maxLen,pvalThreshold=0.05):
    real=__pathFinderCythonChoice__(G,beamwidth,maxLen)

    nullreps=[]
    for G1 in nulls:
        print(len(nullreps), end=' ')
        sys.stdout.flush()
        temp1=__pathFinderCythonChoice__(G1,beamwidth,maxLen)
        nullreps.append([max(y[0] for y in x) if len(x)>0 else 0 for x in temp1])
    nullsRev=list(zip(*nullreps))

    results=[]
    for x in range(len(nullsRev)):
        result=[]
        nullsRev[x]=sorted(nullsRev[x],key=lambda x:-x)
        for item in real[x]:
            count=1
            denom=len(nullsRev[x])+1.0
            for y in nullsRev[x]:
                if item[0]<=y:
                    count+=1
                else:
                    break
            pval=count/denom
            if pval<pvalThreshold:
                result.append([item,pval,stats.norm.ppf(1-pval)])
        results.append(result)
    return results,real,nullsRev
