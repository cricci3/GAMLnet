## This file contains some basic code to perform matrix transforms which can be used by multiple scripts
from numpy import linalg
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import eigsh
from scipy import sparse
import networkx as nx
import numpy as np
import scipy
def getEigs(G):
    if type(G)==nx.Graph:
        eigs=scipy.sparse.linalg.eigsh(nx.to_scipy_sparse_matrix(G,dtype=np.float64,format='csr'),k=2,which='LA',maxiter=100000000000)
    else:
        eigs=scipy.sparse.linalg.eigs(nx.to_scipy_sparse_matrix(G,dtype=np.float64,format='csr'),k=2,which='LR',maxiter=100000000000)
    return eigs

def getEigsSym(G,num=2,nodes=None):
    if nodes:
        A=nx.to_scipy_sparse_matrix(G,dtype=np.float64,format='csr',nodelist=nodes)
    else:
        A=nx.to_scipy_sparse_matrix(G,dtype=np.float64,format='csr')
    if nx.is_directed(G):
        eigs=scipy.sparse.linalg.eigsh(A+A.T,k=num,which='LA',maxiter=100000000000)
        return eigs
    else:
        eigs=scipy.sparse.linalg.eigsh(A,k=num,which='LA',maxiter=100000000000)
        return eigs

def getLaplacianTypes():
    return ['comb', 'rw', 'Lbar', 'Lbar_rw', 'Lbar_sym']


def buildLaplacian(A, typeLap):
    degsA = A.sum(axis=1)
    n = A.shape[1]
    if (degsA==0).sum()>0:
        import pdb
        pdb.set_trace()
    if typeLap=='comb' or typeLap=='combRev':
        if scipy.sparse.issparse(A):
            D=scipy.sparse.spdiags(degsA.T,0,A.shape[0],A.shape[1])
            L =  D - A;
        else:
            D = np.diagflat(degsA);
            L =  D - A;
    elif (typeLap == 'rw'):
        if scipy.sparse.issparse(A):
            Dinv =scipy.sparse.spdiags(1/degsA.T,0,A.shape[0],A.shape[1])
            L =  np.dot(Dinv,A)
        else:
            Dinv = np.diagflat(1/degsA);
            L =  np.dot(Dinv,A)
    elif (typeLap == 'Lbar'):
        print('check orientation (i.e. row sums are are row and not column)')
        absDegs = (abs(A)).sum(axis=1)
        Dbar = np.diagflat(absDegs);
        L = Dbar - A;
    elif ( typeLap == 'Lbar_rw' ):
        print('check orientation (i.e. row sums are are row and not column)')
        absDegs = (abs(A)).sum(axis=1)
        Dbar_inv = diag( 1/absDegs)
        L = np.eye(n)  - np.dot(Dbar_inv, A)
    elif (typeLap == 'Lbar_sym'):
        print('check orientation (i.e. row sums are are row and not column)')
        absDegs = (abs(A)).sum(axis=1)
        Dbar_inv_sqrt = np.diag( absDegs**(-0.5))
        Lbar_sym = np.eye(n) - np.dot(np.dot(Dbar_inv_sqrt,A),Dbar_inv_sqrt)
        L = ( Lbar_sym + Lbar_sym.T ) / 2;
    return L


def timeout_handler(num, stack):
    print("Received SIGALRM")
    raise Exception("FUBAR")

import scipy
def computeTopSpectrum(L,LapType,topk):
    if (LapType== 'comb' or LapType== 'Lbar' or LapType == 'Lbar_sym' ):
        if False: #L.shape[0]<50:
            if type(L)==scipy.sparse.csr.csr_matrix:
                temp1=linalg.eigh(L.todense())
            else:
                temp1=linalg.eigh(L)
            temp2=np.array(temp1[0][:topk])
            temp3=np.array(temp1[1][:,:topk])
            assert(False)
            ansEigs=[temp2,temp3]
        else:
            if topk>=L.shape[0]-1:
                 if scipy.sparse.issparse(L):
                     L=L.todense()
                 ansEigs = np.linalg.eigh(L)
                 ansEigs=list(ansEigs)
                 ansEigs[1]=np.array(ansEigs[1])
            else:
#                 try:
#                     ansEigs = eigsh(L, topk, which = 'SA',maxiter=100000000);
#                     print 'end comb',
#                     import sys
#                     sys.stdout.flush()
#                 except:
#                     ## fall back on to the full decomposition if the sparse version fails
#                     if scipy.sparse.issparse(L):
#                         L=L.todense()
#                     ansEigs = np.linalg.eigh(L)
#                     ansEigs=list(ansEigs)
#                     ansEigs[1]=np.array(ansEigs[1])
#

                 try:
                     ansEigs = eigsh(L, topk, which = 'SA',maxiter=100000000);
                 except Exception as ex:
                     ## fall back on to the full decomposition if the sparse version fails
                     if scipy.sparse.issparse(L):
                         L=L.todense()
                     ansEigs = np.linalg.eigh(L)
                     ansEigs=list(ansEigs)
                     ansEigs[1]=np.array(ansEigs[1])


    elif (LapType== 'combRev' ):   # top eigvals
        ansEigs = eigs(L, topk, which = 'LR');
    elif (LapType== 'rw' ):   # top eigvals
#        if (L>0).sum()>0.1*L.shape[0]*L.shape[1]:
#            ansEigs = eigs(L, topk, which = 'LR');
#        else:
#            ansEigs = eigs(L, topk, which = 'LR');
        if topk>=L.shape[0]-1:
             if scipy.sparse.issparse(L):
                  L=L.todense()
             ansEigs = np.linalg.eig(L)
             ansEigs=list(ansEigs)
             ansEigs[1]=np.array(ansEigs[1])
        else:


            try:
               ansEigs = eigs(L, topk, which = 'LR',maxiter=100000000000)
            except:
                ## fall back on to the full decomposition if the sparse version fails
                if scipy.sparse.issparse(L):
                     L=L.todense()
                ansEigs = np.linalg.eig(L)
                ansEigs=list(ansEigs)
                ansEigs[1]=np.array(ansEigs[1])
    elif ( LapType== 'Lbar_rw' ):
        if (L>0).sum()>0.1*L.shape[0]*L.shape[1]:
            ansEigs = linalg.eigs(L, topk, which = 'SR');
        else:
            ansEigs = eigs(L, topk, which = 'SR');
    ansCompSpect = {}
    ansCompSpect['eigVects'] = ansEigs[1]
    ansCompSpect[ 'eigVals'] = ansEigs[0]
    return ansCompSpect
