import networkx as nx
import random as rd
import numpy as np

cimport numpy as np
DTYPE = np.int
DTYPE1 = np.double

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int_t DTYPE_t
ctypedef np.double_t DTYPE1_t

cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)

def directedConfigModelMk3Gen(indegC,outdegC,weightsC):
    cdef int n = len(indegC)
    cdef int e=len(weightsC)
    cdef int x;
    cdef int y;
    cdef int z=0;
    cdef np.ndarray[DTYPE_t, ndim=1]  indeg=np.zeros([e,],dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1]  indegC1=np.array(indegC,dtype=DTYPE)
    for x in range(n):
        for y in range(indegC1[x]):
            indeg[z]=x
            z+=1
    z=0
    cdef np.ndarray[DTYPE_t, ndim=1]  outdeg=np.zeros([e,],dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1]  outdegC1=np.array(outdegC,dtype=DTYPE)
    for x in range(n):
        for y in range(outdegC1[x]):
            outdeg[z]=x
            z+=1
    cdef np.ndarray[DTYPE1_t, ndim=1] weights=np.array(weightsC,dtype=DTYPE1)
    np.random.shuffle(indeg)
    np.random.shuffle(weights)
    cdef list G1=[]
    G1=[{} for x in range(len(indegC))]
    for i in range(e):
        G1[outdeg[i]][indeg[i]]=weights[i]
    for i in range(n):
        if i in G1[i]:
            del G1[i][i]
    return G1


def stripDeg0NodesSimpleGraph(G1):
    n=len(G1)
    toExclude=[]
    inDegree=[0 for x in range(n)]
    for i in range(n):
        for j in G1[i]:
            inDegree[j]+=1
    for i in range(n):
        if len(G1[i])==0 and inDegree[i]==0:
            toExclude.append(i)
    if len(toExclude)==0:
        return G1
    count=0
    remap=list(range(toExclude[0]))
    toExclude.append(n+1)
    excluded=0
    for j in range(len(toExclude)-1):
        remap.append(-1)
        excluded+=1
        remap+=range(toExclude[j]+1-excluded,toExclude[j+1]-excluded)
    G2=[{} for x in range(n-len(toExclude)+1)]
    for i in range(n):
        for j in G1[i]:
            G2[remap[i]][remap[j]]=G1[i][j]
    return G2

