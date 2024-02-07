import networkx as nx
import itertools as it

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

## ordering as in milo 2004
def motifCounterFastProductSimpleGraph(G):
    cdef int n =len(G)
    #result=[[0 for x in range(14)] for y in range(len(G))]
    result=[]
    cdef np.ndarray[DTYPE1_t, ndim=1] motif1=np.zeros([n,],dtype=DTYPE1)
    cdef np.ndarray[DTYPE1_t, ndim=1] motif2=np.zeros([n,],dtype=DTYPE1)
    cdef np.ndarray[DTYPE1_t, ndim=1] motif3=np.zeros([n,],dtype=DTYPE1)
    cdef np.ndarray[DTYPE1_t, ndim=1] motif4=np.zeros([n,],dtype=DTYPE1)
    cdef np.ndarray[DTYPE1_t, ndim=1] motif5=np.zeros([n,],dtype=DTYPE1)
    cdef np.ndarray[DTYPE1_t, ndim=1] motif6=np.zeros([n,],dtype=DTYPE1)
    cdef np.ndarray[DTYPE1_t, ndim=1] motif7=np.zeros([n,],dtype=DTYPE1)
    cdef np.ndarray[DTYPE1_t, ndim=1] motif8=np.zeros([n,],dtype=DTYPE1)
    cdef np.ndarray[DTYPE1_t, ndim=1] motif9=np.zeros([n,],dtype=DTYPE1)
    cdef np.ndarray[DTYPE1_t, ndim=1] motif10=np.zeros([n,],dtype=DTYPE1)
    cdef np.ndarray[DTYPE1_t, ndim=1] motif11=np.zeros([n,],dtype=DTYPE1)
    cdef np.ndarray[DTYPE1_t, ndim=1] motif12=np.zeros([n,],dtype=DTYPE1)
    cdef np.ndarray[DTYPE1_t, ndim=1] motif13=np.zeros([n,],dtype=DTYPE1)
    result.append(motif1)
    result.append(motif2)
    result.append(motif3)
    result.append(motif4)
    result.append(motif5)
    result.append(motif6)
    result.append(motif7)
    result.append(motif8)
    result.append(motif9)
    result.append(motif10)
    result.append(motif11)
    result.append(motif12)
    result.append(motif13)
    cdef int x
    cdef int y
    cdef int n1
    cdef int n2
    cdef double w1
    cdef double w2
    cdef double w3
    cdef double w4
    preS=[set() for x in range(n)]
    for x in range(n):
        for y in G[x]:
            preS[y].add(x)
    for x in range(n):
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
                motif10[x]  +=w1*w2*w3*w4 
                motif10[n1]+=w1*w2*w3*w4
                motif10[n2]+=w1*w2*w3*w4
#                result[x][10]+=w1*w2*w3*w4
#                result[n1][10]+=w1*w2*w3*w4
#                result[n2][10]+=w1*w2*w3*w4
            elif (c1):
                w3=G[n1][n2]
                motif7[x]+=w1*w2*w3
                motif7[n1]+=w1*w2*w3
                motif7[n2]+=w1*w2*w3
#                result[x][7]+=w1*w2*w3
#                result[n1][7]+=w1*w2*w3
#                result[n2][7]+=w1*w2*w3
            elif (c2):
                w3=G[n2][n1]
                motif7[x]+=w1*w2*w3
                motif7[n1]+=w1*w2*w3
                motif7[n2]+=w1*w2*w3
#                result[x][7]+=w1*w2*w3
#                result[n1][7]+=w1*w2*w3
#                result[n2][7]+=w1*w2*w3
        for n1,n2 in it.combinations(pre,2):
            w1=G[n1][x]
            w2=G[n2][x]
            c1=n2 in G[n1]
            c2=n1 in G[n2]
            if (not c1) and (not c2):
                motif2[x]+=w1*w2
                motif2[n1]+=w1*w2
                motif2[n2]+=w1*w2
#                result[x][2]+=w1*w2
#                result[n1][2]+=w1*w2
#                result[n2][2]+=w1*w2
        for n1,n2 in it.combinations(suc0,2):
            w1=G[x][n1]
            w2=G[x][n2]
            c1=n2 in G[n1]
            c2=n1 in G[n2]
            if (c1 and c2):
                w3=G[n1][n2]
                w4=G[n2][n1]
                motif9[x] +=w1*w2*w3*w4
                motif9[n1]+=w1*w2*w3*w4
                motif9[n2]+=w1*w2*w3*w4
#                result[x][9] +=w1*w2*w3*w4
#                result[n1][9]+=w1*w2*w3*w4
#                result[n2][9]+=w1*w2*w3*w4
            elif (c1):
                w3=G[n1][n2]
                motif7[x] +=w1*w2*w3
                motif7[n1]+=w1*w2*w3
                motif7[n2]+=w1*w2*w3
#                result[x][7] +=w1*w2*w3
#                result[n1][7]+=w1*w2*w3
#                result[n2][7]+=w1*w2*w3
            elif (c2):
                w3=G[n2][n1]
                motif7[x] +=w1*w2*w3
                motif7[n1]+=w1*w2*w3
                motif7[n2]+=w1*w2*w3
#                result[x][7] +=w1*w2*w3
#                result[n1][7]+=w1*w2*w3
#                result[n2][7]+=w1*w2*w3
        for n1,n2 in it.combinations(suc,2):
            w1=G[x][n1]
            w2=G[x][n2]
            c1=n2 in G[n1]
            c2=n1 in G[n2]
            if (not c1) and (not c2):
                motif1[x] +=w1*w2
                motif1[n1]+=w1*w2
                motif1[n2]+=w1*w2
#                result[x][1] +=w1*w2
#                result[n1][1]+=w1*w2
#                result[n2][1]+=w1*w2
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
                motif13[x] +=w1*w2*w3*w4
                motif13[n1]+=w1*w2*w3*w4
                motif13[n2]+=w1*w2*w3*w4
#                result[x][13] +=w1*w2*w3*w4
#                result[n1][13]+=w1*w2*w3*w4
#                result[n2][13]+=w1*w2*w3*w4
            elif (c1):
                w3=G[n1][n2]
                motif12[x] +=w1*w2*w3
                motif12[n1]+=w1*w2*w3
                motif12[n2]+=w1*w2*w3
#                result[x][12] +=w1*w2*w3
#                result[n1][12]+=w1*w2*w3
#                result[n2][12]+=w1*w2*w3
            elif (c2):
                w3=G[n2][n1]
                motif12[x] +=w1*w2*w3
                motif12[n1]+=w1*w2*w3
                motif12[n2]+=w1*w2*w3
#                result[x][12] +=w1*w2*w3
#                result[n1][12]+=w1*w2*w3
#                result[n2][12]+=w1*w2*w3
        for n1,n2 in it.combinations(both,2):
            w1= G[x][n1]
            w2= G[x][n2]
            w1*=G[n1][x]
            w2*=G[n2][x]
            c1=n2 in G[n1]
            c2=n1 in G[n2]
            if (not c1) and (not c2):
                motif6[x] +=w1*w2
                motif6[n1]+=w1*w2
                motif6[n2]+=w1*w2
#                result[x][6] +=w1*w2
#                result[n1][6]+=w1*w2
#                result[n2][6]+=w1*w2
        for n1 in pre0:
            w1=G[n1][x]
            for n2 in suc0:
                w2=G[x][n2]
                c1=n2 in G[n1]
                c2=n1 in G[n2]
                if (c1 and c2):
                    w3=G[n1][n2]
                    w4=G[n2][n1]
                    motif11[x] +=w1*w2*w3*w4
                    motif11[n1]+=w1*w2*w3*w4
                    motif11[n2]+=w1*w2*w3*w4
#                    result[x][11] +=w1*w2*w3*w4
#                    result[n1][11]+=w1*w2*w3*w4
#                    result[n2][11]+=w1*w2*w3*w4
                elif (c1):
                    w3=G[n1][n2]
                    motif7[x] +=w1*w2*w3
                    motif7[n1]+=w1*w2*w3
                    motif7[n2]+=w1*w2*w3
#                    result[x][7] +=w1*w2*w3
#                    result[n1][7]+=w1*w2*w3
#                    result[n2][7]+=w1*w2*w3
                elif (c2):
                    w3=G[n2][n1]
                    motif8[x] +=w1*w2*w3
                    motif8[n1]+=w1*w2*w3
                    motif8[n2]+=w1*w2*w3
#                    result[x][8]+=w1*w2*w3
#                    result[n1][8]+=w1*w2*w3
#                    result[n2][8]+=w1*w2*w3
        for n1 in pre:
            w1=G[n1][x]
            for n2 in suc:
                c1=n2 in G[n1]
                c2=n1 in G[n2]
                if (not c1) and (not c2):
                    w2=G[x][n2]
                    motif3[x] +=w1*w2
                    motif3[n1]+=w1*w2
                    motif3[n2]+=w1*w2
#                    result[x][3] +=w1*w2
#                    result[n1][3]+=w1*w2
#                    result[n2][3]+=w1*w2
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
                    motif12[x] +=w1*w2*w3*w4
                    motif12[n1]+=w1*w2*w3*w4
                    motif12[n2]+=w1*w2*w3*w4
#                    result[x][12] +=w1*w2*w3*w4
#                    result[n1][12]+=w1*w2*w3*w4
#                    result[n2][12]+=w1*w2*w3*w4
                elif (c1):
                    w3=G[n1][n2]
                    motif9[x] +=w1*w2*w3
                    motif9[n1]+=w1*w2*w3
                    motif9[n2]+=w1*w2*w3
#                    result[x][9] +=w1*w2*w3
#                    result[n1][9]+=w1*w2*w3
#                    result[n2][9]+=w1*w2*w3
                elif (c2):
                    w3=G[n2][n1]
                    motif11[x] +=w1*w2*w3
                    motif11[n1]+=w1*w2*w3
                    motif11[n2]+=w1*w2*w3
#                    result[x][11] +=w1*w2*w3
#                    result[n1][11]+=w1*w2*w3
#                    result[n2][11]+=w1*w2*w3
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
                    motif12[x] +=w1*w2*w3*w4
                    motif12[n1]+=w1*w2*w3*w4
                    motif12[n2]+=w1*w2*w3*w4
#                    result[x][12] +=w1*w2*w3*w4
#                    result[n1][12]+=w1*w2*w3*w4
#                    result[n2][12]+=w1*w2*w3*w4
                elif (c1):
                    w3=G[n1][n2]
                    motif11[x] +=w1*w2*w3
                    motif11[n1]+=w1*w2*w3
                    motif11[n2]+=w1*w2*w3
#                    result[x][11] +=w1*w2*w3
#                    result[n1][11]+=w1*w2*w3
#                    result[n2][11]+=w1*w2*w3
                elif (c2):
                    w3=G[n2][n1]
                    motif10[x] +=w1*w2*w3
                    motif10[n1]+=w1*w2*w3
                    motif10[n2]+=w1*w2*w3
#                    result[x][10] +=w1*w2*w3
#                    result[n1][10]+=w1*w2*w3
#                    result[n2][10]+=w1*w2*w3

        for n2 in both:
            w2=G[x][n2]
            w2*=G[n2][x]
            for n1 in suc:
                w1=G[x][n1]
                c1=n2 in G[n1]
                c2=n1 in G[n2]
                if (not c1) and (not c2):
                    motif5[x] +=w1*w2
                    motif5[n1]+=w1*w2
                    motif5[n2]+=w1*w2
#                    result[x][5] +=w1*w2
#                    result[n1][5]+=w1*w2
#                    result[n2][5]+=w1*w2
            for n1 in pre:
                w1=G[n1][x]
                c1=n2 in G[n1]
                c2=n1 in G[n2]
                if (not c1) and (not c2):
                    motif4[x] +=w1*w2
                    motif4[n1]+=w1*w2
                    motif4[n2]+=w1*w2
#                    result[x][4] +=w1*w2
#                    result[n1][4]+=w1*w2
#                    result[n2][4]+=w1*w2
    result1=list(zip(*result))
    # result1=zip(*result)
    return result1
