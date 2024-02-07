import numpy as np
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2 import robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
orca=importr("orca")
igraph = importr('igraph')
netdist =importr('netdist')
countsToGdd=rpy2.robjects.r['graph_features_to_histograms']

def getGDD(data):
    data=np.array([np.array(x) for x in data])
    Bm=robjects.r.matrix(data,nrow=data.shape[0],ncol=data.shape[1])
    sdf=countsToGdd(Bm)
    return sdf

def comparePairOfGdds(x,y):
    return netdist.net_emd(x,y)
