import heapq

## tested
def __pathFinderHelper__(G,beamwidth,maxLen,returnAll=True):
    Grev=[{} for x in range(len(G))]
    for x in range(len(G)):
        for y in G[x]:
            Grev[y][x]=G[x][y]
    Q=[[0,[]] for x in range(beamwidth)]
    heapq.heapify(Q)
    for x in range(len(G)):
        for y in G[x]:
            pop1=heapq.heappushpop(Q,[G[x][y],[x,y]])
    Q=[[0,[]] for x in range(beamwidth)]
    for x in range(len(G)):
        t1=sorted(list(G[x].items()),key=lambda x:x[1])
        if len(t1)==0:
            continue
        t2=sorted(list(Grev[x].items()),key=lambda x:x[1])
        if len(t2)==0:
            continue
        t1s=t1.pop()
        t2s=t2.pop()
        while (min(t1s[1],t2s[1])>Q[0][0]):
            t3=min(t1s[1],t2s[1])
            pop1=heapq.heappushpop(Q,[t3,[t2s[0],x,t1s[0]]])
            if len(t1)==0:
                 if len(t2)==0:
                     break
                 else:
                     t2s=t2.pop()
            else:
                 if len(t2)==0:
                     t1s=t1.pop()
                 else:
                     if t1[-1][1]>t2[-1][1]:
                           t1s=t1.pop()
                     else:
                           t2s=t2.pop()
    Q.sort(key=lambda x:-x[0])
    result=[[x for x in Q if x[0]>0],]
    for curLen in range(maxLen-2):
        newQ=[]
        newQ=[[0,[]] for x in range(beamwidth)]
        heapq.heapify(newQ)
        m1=0
        count=0
        count1=0
        for item in Q:
            if len(item[1])==0:
                continue
            if item[0]<newQ[0][0]:
                count+=1
                continue
            for x in G[item[1][-1]]:
                if x in item[1]:
                    continue
                t1=G[item[1][-1]][x]
                if item[0]<t1:
                    temp1=[item[0],item[-1]+[x,]]
                else:
                    if t1<newQ[0][0]:
                        continue
                    temp1=[t1,item[-1]+[x,]]
                pop1=heapq.heappushpop(newQ,temp1)
                count1+=1
        s1=sum([1 for x in newQ if x[0]==0])
        if s1==len(Q):
            assert(False)
        Q=newQ
        Q=[x for x in Q if x[0]>0]
        assert(len(Q)==min(count1,beamwidth))
        Q.sort(key=lambda x:-x[0])
        if returnAll:
            result.append(Q)
    if returnAll:
        return result
    else:
        return Q

