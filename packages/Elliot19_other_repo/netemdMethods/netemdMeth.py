import fraudapp.detector.netemdMethods.netemdFuncs as nemd
import fraudapp.detector.netemdMethods.countOrbits.countOrbitsCython as coc
import sys
sys.path.append('../')
import fraudapp.detector.configModel.configurationModel as cm
import numpy as np
from scipy import stats

def measureDistance(referenceGraphsGDDs,observationGDD):
    distances = []
    for item in referenceGraphsGDDs:
        distances1=[]
        for i in range(len(item)):
            currentDist = float(nemd.comparePairOfGdds(item[i], observationGDD[i])[0])
            distances1.append(currentDist)
        distances.append(distances1)
    return distances

def measureMotifStats(G,numReps=100):
    configGen = cm.simpleConfigModelGen(G)
    if sys.version_info[0] >= 3:
        referenceGraphs = [configGen.__next__() for x in range(15)]
    else:
        referenceGraphs = [configGen.next() for x in range(15)]

    motifFind = coc.motifCounterFastProductSimpleGraph
    referMotifs = [motifFind(x) for x in referenceGraphs]
    referGdds = [nemd.getGDD(np.array(x)) for x in referMotifs]
    configMotifs = (motifFind(x) for x in configGen)
    configGdds = (nemd.getGDD(np.array(x)) for x in configMotifs)
    distances=[]
    for i in range(numReps):
        if sys.version_info[0] >= 3:
            temp1=measureDistance(referGdds,configGdds.__next__())
        else:
            temp1=measureDistance(referGdds,configGdds.next())
        dists=stats.trim_mean(np.array(temp1),0.1)
        distances.append(dists)

    # Real observation
    realMotifs=motifFind(G)
    realGdds=nemd.getGDD(np.array(realMotifs))
    temp1=measureDistance(referGdds,realGdds)
    realdists=stats.trim_mean(np.array(temp1),0.1)
    results={}
    for motif in range(13):
        numerator=1
        denominator=float(1+numReps)
        for i in range(numReps):
            if realdists[motif]<=distances[i][motif]:
                numerator+=1
        pval=numerator/denominator
        print(pval)
        if pval<0.05:
            obs = [x[motif] for x in realMotifs]
            m1=np.mean(obs)
            std=np.std(obs)

            tempRes={}
            results['motif'+str(motif)] = tempRes
            for j in range(len(G)):
                temp=(obs[j]-m1)/std
                temp=abs(temp)
                if temp>=2:
                    tempRes[j] = temp
                else:
                    tempRes[j] = 0
        else:
            results['motif'+str(motif)] = {j:0 for j in range(len(G))}
    return results
