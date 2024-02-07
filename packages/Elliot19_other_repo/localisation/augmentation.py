from collections import Counter
from collections import defaultdict
from math import ceil
from scipy import stats
import networkx as nx
import random as rd
import time
import time
try:
    import numpy as np
except:
    print('no numpy')

# tested
def modNetworkRep(G,thresPercent=0.99,inPlace=False):
    #weights=np.array([G.edge[x][y]['weight'] for x in G for y in G.edge[x]])
    #threshold=np.percentile(weights,thresPercent)
    threshold=__getThreshold__(G,thresPercent)
    n1=G.number_of_edges()
    if inPlace==False:
        G=G.copy()
    G,change=modNetwork(G,threshold=threshold,inPlace=True,reportChanges=True)
    count=0
    while (change):
        print(count)
        G,change=modNetwork(G,threshold=threshold,inPlace=True,reportChanges=True)
        #G,change=modNetwork(G,thresPercent,False,reportChanges=True)
        count+=1
    return G

def __getThreshold__(G,thresPercent=0.99):
     weights=np.array([G[x][y]['weight'] for x in G for y in G[x]])
     threshold=np.percentile(weights,thresPercent*100)
     return threshold


# tested
def modNetwork(G,thresPercent=0.99,threshold=None,inPlace=True,reportChanges=False):
     if threshold==None:
        threshold=__getThreshold__(G,thresPercent)
     print(threshold)
     if inPlace:
         G1=G
     else:
         G1=G.copy()
     toAdd=[]
     count=0
     for x in G:
         for y in G[x]:
 #            count+=1
 #            if count%100==0:
 #                print count/float(G.number_of_edges())
             if G[x][y]['weight']>=threshold:
                 for z in G[y]:
                     if z==x:
                         continue
                     if G[y][z]['weight']>=threshold:
                         m1=G[x][y]['weight']
                         m2=G[y][z]['weight']
                         m3=min(m1,m2)
                         if z in G[x]:
                             if m3<G[x][z]['weight']:
                                 continue
                         toAdd.append([x,z,min(m1,m2)])
     changed=False
     for item in toAdd:
           if item[1] in G1[item[0]]:
                w1=G1[item[0]][item[1]]['weight']
                if item[2]>w1:
                    G1[item[0]][item[1]]['weight']=max(item[2],w1)
#                    G1.add_edge(item[0],item[1],weight=max(item[2],w1))
                    changed=True
           else:
                G1.add_edge(item[0],item[1],weight=item[2])
                changed=True
     if reportChanges:
         return G1,changed
     else:
         return G1

