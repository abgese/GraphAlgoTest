import numpy as np
import random

class Graph:
    def __init__(self, numVertices, maxEdgeWeight=100,edgeProb=None, avgNumEdges=None, noEdges=False):
        self.numVertices = numVertices
        self.maxEdgeWeight = maxEdgeWeight

        
        self.E = [[] for i in range(self.numVertices)]
        #Create Graph with no Edges
        if noEdges:
            return

        # Ensure graph is connected
        for i in range(numVertices):
            edgeWeight = random.randint(1, self.maxEdgeWeight)
            neighbour = (i+1) % self.numVertices
            self.AddEdge((i,neighbour,edgeWeight))

        self.edgeProb = edgeProb
        self.avgNumEdges = avgNumEdges

        if(self.avgNumEdges != None):
            numEdgeSelect = range(2 * self.avgNumEdges) # Number of edges will be sample from int range 0 to 2*avgNumEdges, thus on average there will be avgNumEdges

            for i in range(self.numVertices):
                edgeSelect = np.random.rand(self.numVertices)
                edgeSelect[i] = 0.0
                for w in self.E[i]:
                    edgeSelect[w] = 0.0 # remove edges already present from possible choice

                numEdges = np.random.choice(numEdgeSelect)
                numEdges = max(numEdges - len(self.E[i]), 0)
                edgeSelectArgSort = np.argsort(-edgeSelect)[:numEdges]

                for w in edgeSelectArgSort:
                    edgeWeight = random.randint(1, self.maxEdgeWeight) 
                    self.AddEdge((i,w,edgeWeight))

        elif(self.edgeProb != None):
            for i in range(self.numVertices - 2):
                # Consider every edge only once to ensure correct probability of picking
                edgeSelect = np.random.rand(self.numVertices - 2 - i) 
                for j in range(len(edgeSelect)):
                    # Considers all vertices from i+2 to numVertices - 1 (i+1 is already connected). Also for i == 0 the last vertex is ignored
                    if(edgeSelect[j] <= self.edgeProb not (i+2+j == self.numVertices - 1 and i==0)):
                        edgeWeight = random.randint(1, self.maxEdgeWeight) 
                        self.AddEdge((i,i+2+j,edgeWeight)) 

        else:
            raise Exception("Requires Either average number of edges per vertex or percentage of edges")

    def AddEdge(self, edge):
        if(edge[0] < 0 or edge[0] >= self.numVertices or edge[1] < 0 or edge[1] >= self.numVertices):
            raise Exception("Trying to add edge for vertices that don't exist")
        self.E[edge[0]].append((edge[1],edge[2]))
        self.E[edge[1]].append((edge[0],edge[2]))

class HeapNode:
    def __init__(self, key, weight):
        self.key = key
        self.weight = weight

class MaxHeap:
    def __init__(self):
       self.heapElements = []
       self.numElements = 0

    def Top(self):
       return self.heapElements[0]
       
    def Delete(self, i, heapIndex=[]):
        if(i >= self.numElements):
            raise Exception("Cannot delete element outside of range of Heap")
        self.heapElements[i], self.heapElements[self.numElements - 1] = self.heapElements[self.numElements - 1], self.heapElements[i] # Swap i'th value with last value
        if(len(heapIndex) > 0):
            #swap heap index values
            heapIndex[self.heapElements[i].key], heapIndex[self.heapElements[self.numElements - 1].key] = heapIndex[self.heapElements[self.numElements - 1].key], heapIndex[self.heapElements[i].key]
        self.heapElements.pop()
        self.numElements -= 1

        parent = max(int((i-1)/2),0)
        while(i > 0 and self.heapElements[i].weight > self.heapElements[parent].weight):
            self.heapElements[i], self.heapElements[parent] = self.heapElements[parent], self.heapElements[i]
            if(len(heapIndex) > 0):
                #swap heap index values
                heapIndex[self.heapElements[i].key], heapIndex[self.heapElements[parent].key] = heapIndex[self.heapElements[parent].key], heapIndex[self.heapElements[i].key]
            i = parent
            parent = max(int((i-1)/2),0)

        while(2*i + 1 < self.numElements):
            maxVal = self.heapElements[i]
            maxInd = i
            if(maxVal.weight < self.heapElements[2*i + 1].weight):
                maxVal = self.heapElements[2*i + 1]
                maxInd = 2*i + 1

            if(2*i + 2 < self.numElements and maxVal.weight< self.heapElements[2*i + 2].weight):
                maxVal = self.heapElements[2*i + 2]
                maxInd = 2*i + 2

            if(maxInd == i):
                break

            else:
                self.heapElements[i], self.heapElements[maxInd] = self.heapElements[maxInd], self.heapElements[i]
                if(len(heapIndex) > 0):
                    #swap heap index values
                    heapIndex[self.heapElements[i].key], heapIndex[self.heapElements[maxInd].key] = heapIndex[self.heapElements[maxInd].key], heapIndex[self.heapElements[i].key]
                i = maxInd

    def Insert(self, val, heapIndex = []):
        self.heapElements.append(val)
        self.numElements += 1
        i = self.numElements - 1
        if(len(heapIndex) > 0):
            heapIndex[val.key] = i

        parent = max(int((i-1)/2),0)
        while(i > 0 and self.heapElements[i].weight > self.heapElements[parent].weight):
            self.heapElements[i], self.heapElements[parent] = self.heapElements[parent], self.heapElements[i]
            if(len(heapIndex) > 0):
                heapIndex[self.heapElements[i].key], heapIndex[self.heapElements[parent].key] = heapIndex[self.heapElements[parent].key], heapIndex[self.heapElements[i].key]
            i = parent
            parent = max(int((i-1)/2),0)

        return i

class UnionFind:
    def __init__(self, numVertices):
        self.numVertices = numVertices
        self.parent = np.zeros(self.numVertices, dtype=int)
        self.rank = np.zeros(self.numVertices ,dtype=int)

    def Union(self, i, j):
        parent_i = self.Find(i)
        parent_j = self.Find(j)
        if(self.rank[parent_i] > self.rank[parent_j]):
            self.parent[parent_j] = parent_i
        elif(self.rank[parent_j] > self.rank[parent_i]):
            self.parent[parent_i] = parent_j
        else:
            self.parent[parent_i] = parent_j
            self.rank[parent_j] += 1

    def Find(self, i):
        if(self.parent[i] != i and self.parent[i] != 0):
            self.parent[i] = self.Find(self.parent[i])
        return self.parent[i]

    def MakeSet(self, i):
        self.parent[i] = i
        self.rank[i] = 1


def HeapSort(arr):
    H = MaxHeap()
    for val in arr:
        H.Insert(val)

    sortedArr = []
    for i in range(len(arr)):
        sortedArr.append(H.Top())
        H.Delete(0)

    return sortedArr

UNSEEN = 0
FRINGE = 1
INTREE = 2
def MBWDjikstraNoHeap(G, s, t):
    status = np.zeros(G.numVertices, dtype=int)
    bandwidth = np.zeros(G.numVertices)
    parent = np.array([ -1 for i in range(G.numVertices)])
    status[s] = INTREE
    numFringes = 0
    for w,wt in G.E[s]:
        status[w] = FRINGE
        bandwidth[w] = wt
        parent[w] = s
        numFringes += 1

    while(numFringes > 0):
        maxVal = bandwidth[0]
        maxIndex = 0
        numFringes -= 1
        for i in range(G.numVertices):
            if(status[i] == FRINGE and maxVal < bandwidth[i]):
                maxVal = bandwidth[i]
                maxIndex = i

        status[maxIndex] = INTREE
        for w,wt in G.E[maxIndex]:
            if status[w] == UNSEEN:
                status[w] = FRINGE
                parent[w] = maxIndex
                bandwidth[w] = min(bandwidth[maxIndex], wt)
                numFringes += 1

            elif status[w] == FRINGE:
                if(bandwidth[w] < min(bandwidth[maxIndex], wt)):
                    parent[w] = maxIndex
                    bandwidth[w] = min(bandwidth[maxIndex], wt)

    return bandwidth, parent


def MBWDjikstraHeap(G, s, t):
    status = np.zeros(G.numVertices, dtype=int)
    bandwidth = np.zeros(G.numVertices)
    parent = np.array([ -1 for i in range(G.numVertices)])
    heapIndex = np.array([ -1 for i in range(G.numVertices)])
    status[s] = INTREE
    fringeHeap = MaxHeap()
    for w,wt in G.E[s]:
        status[w] = FRINGE
        bandwidth[w] = wt
        parent[w] = s
        fringeHeap.Insert(HeapNode(w,bandwidth[w]),heapIndex=heapIndex)

    while(fringeHeap.numElements > 0):
        maxVal = fringeHeap.Top()
        maxIndex = maxVal.key
        status[maxIndex] = INTREE
        fringeHeap.Delete(0, heapIndex=heapIndex)
        for w,wt in G.E[maxIndex]:
            if status[w] == UNSEEN:
                status[w] = FRINGE
                parent[w] = maxIndex
                bandwidth[w] = min(bandwidth[maxIndex], wt)
                fringeHeap.Insert(HeapNode(w,bandwidth[w]),heapIndex=heapIndex)

            elif status[w] == FRINGE:
                if(bandwidth[w] < min(bandwidth[maxIndex], wt)):
                    parent[w] = maxIndex
                    bandwidth[w] = min(bandwidth[maxIndex], wt)
                    fringeHeap.Delete(heapIndex[w], heapIndex=heapIndex)
                    fringeHeap.Insert(HeapNode(w,bandwidth[w]),heapIndex=heapIndex)

    return bandwidth, parent

def MBWinTree(G, s, t): # DFS -> Greedy works since no more than one path between two vertices in Tree
    status = np.zeros(G.numVertices, dtype=int)
    bandwidth = np.zeros(G.numVertices)
    parent = np.array([ -1 for i in range(G.numVertices)])
    vertexStack = [] #Using stack for iterative form of DFS
    status[s] = 1

    for w,wt in G.E[s]:
        vertexStack.append(w)
        bandwidth[w] = wt
        status[w] = 1
        parent[w] = s

    while(len(vertexStack) > 0):
        v = vertexStack[-1] # Top of stack
        vertexStack.pop()
        for w,wt in G.E[v]:
            if(status[w] != 1):
                vertexStack.append(w)
                bandwidth[w] = min(bandwidth[v], wt)
                status[w] = 1
                parent[w] = v

    return bandwidth, parent


def MBWKruskal(G, s, t):
    edgeList = []
    subTrees = UnionFind(G.numVertices)
    maxSpanTree = Graph(G.numVertices,noEdges=True) #Graph with no edges on init
    bandwidth = np.zeros(G.numVertices)

    for i in range(G.numVertices):
        for w,wt in G.E[i]:
            if(w>i): # Ensure only unique edges considered
                edgeList.append(HeapNode(key=(i,w),weight=wt))
                
    edgeList = HeapSort(edgeList)
    for edge in edgeList:
        if(subTrees.Find(edge.key[0]) == 0):
            subTrees.MakeSet(edge.key[0])
        if(subTrees.Find(edge.key[1]) == 0):
            subTrees.MakeSet(edge.key[1])
        if(subTrees.Find(edge.key[0]) != subTrees.Find(edge.key[1])):
            maxSpanTree.AddEdge((edge.key[0],edge.key[1],edge.weight))
            subTrees.Union(edge.key[0],edge.key[1])

    # return maxSpanTree
    return MBWinTree(maxSpanTree, s, t)
