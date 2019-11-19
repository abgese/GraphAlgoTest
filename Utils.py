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
            self.E[i].append((neighbour,edgeWeight))
            self.E[neighbour].append((i, edgeWeight))

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
                    self.E[i].append((w, edgeWeight))
                    self.E[w].append((i, edgeWeight)

        elif(self.edgeProb != None):
            for i in range(self.numVertices - 2):
                edgeSelect = np.random.rand(self.numVertices - 2 - i)
                for j in range(len(edgeSelect)):
                    if((i + 2 + j) < self.numVertices and edgeSelect[j] <= self.edgeProb):
                        self.E[i].append(i+2+j)
                        self.E[i+2+j].append(i)

        else:
            raise Exception("Requires Either average number of edges per vertex or percentage of edges")

    def AddEdge(self, edge):
        if(edge[0] < 0 or edge[0] >= self.numVertices or edge[1] < 0 or edge[1] >= self.numVertices):
            raise Exception("Trying to add edge for vertices that don't exist")
        self.E[edge[0]].append((edge[1],edge[2]))
        self.E[edge[1]].append((edge[0],edge[2]))

class MaxHeap:
    def __init__(self, weightIndex):
       self.heapElements = []
       self.numElements = 0
       self.weightIndex = weightIndex # Position of weight in tuple being stored e.g. (v, d[v]) in case of Djikstra's

    def Top(self):
       return self.heapElements[0]
       
    def Delete(self, i):
        if(i >= self.numElements):
            raise Exception("Cannot delete element outside of range of Heap")
        self.heapElements[i], self.heapElements[self.numElements - 1] = self.heapElements[self.numElements - 1], self.heapElements[i] # Swap i'th value with last value
        self.heapElements.pop()
        self.numElements -= 1
        if(i == 0):
            return
        parent = max(int((i-1)/2),0)
        while(self.heapElements[i][self.weightIndex] > self.heapElements[parent][self.weightIndex]):
            self.heapElements[i], self.heapElements[parent] = self.heapElements[parent], self.heapElements[i]
            i = parent
            parent = max(int((i-1)/2),0)

        while(2*i + 1 < self.numElements):
            maxVal = self.heapElements[i]
            maxInd = i
            if(maxVal[self.weightIndex] < self.heapElements[2*i + 1][self.weightIndex]):
                maxVal = self.heapElements[2*i + 1]
                maxInd = 2*i + 1

            if(2*i + 2 < self.numElements and maxVal[self.weightIndex] < self.heapElements[2*i + 2][self.weightIndex]):
                maxVal = self.heapElements[2*i + 2]
                maxInd = 2*i + 2

            if(maxInd == i):
                break
            else:
                self.heapElements[i], self.heapElements[maxInd] = self.heapElements[maxInd], self.heapElements[i]
                i = maxInd

    def Insert(self, val):
        self.heapElements.append(val)
        self.numElements += 1
        i = self.numElements - 1
        parent = max(int((i-1)/2),0)
        while(self.heapElements[i][self.weightIndex] > self.heapElements[parent][self.weightIndex]):
            self.heapElements[i], self.heapElements[parent] = self.heapElements[parent], self.heapElements[i]
            i = parent
            parent = max(int((i-1)/2),0)

        return i


def HeapSort(arr, weightIndex):
    H = MaxHeap(weightIndex=weightIndex)
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
    status = np.zeros(G.numVertices)
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
        for i in range(G.numVertices):
            if(status[i] == FRINGE and maxVal < bandwidth[i]):
                maxVal = bandwidth[i]
                maxIndex = maxVal

        status[maxIndex] = INTREE
        for w,wt in G.E[maxIndex]:
            if status[w] == UNSEEN:
                status[w] = FRINGE
                parent[w] = maxIndex
                bandwidth[w] = min(bandwidth[maxIndex], wt)

            elif status[w] == FRINGE:
                if(bandwidth[w] < min(bandwidth[maxIndex], wt)):
                    parent[w] = maxIndex
                    bandwidth[w] = min(bandwidth[maxIndex], wt)

    return bandwidth, parent


def MBWDjikstraHeap(G, s, t):
    status = np.zeros(G.numVertices)
    bandwidth = np.zeros(G.numVertices)
    parent = np.array([ -1 for i in range(G.numVertices)])
    heapIndex = np.array([ -1 for i in range(G.numVertices)])
    status[s] = INTREE
    fringeHeap = MaxHeap(weightIndex=1)
    for w,wt in G.E[s]:
        status[w] = FRINGE
        bandwidth[w] = wt
        parent[w] = s
        heapIndex[w] = fringeHeap.Insert((w,bandwidth[w]))

    while(fringeHeap.numElements > 0):
        maxIndex, maxVal = fringeHeap.Top()   

        status[maxIndex] = INTREE
        for w,wt in G.E[maxIndex]:
            if status[w] == UNSEEN:
                status[w] = FRINGE
                parent[w] = maxIndex
                bandwidth[w] = min(bandwidth[maxIndex], wt)
                heapIndex[w] = fringeHeap.Insert((w,bandwidth[w]))

            elif status[w] == FRINGE:
                if(bandwidth[w] < min(bandwidth[maxIndex], wt)):
                    parent[w] = maxIndex
                    bandwidth[w] = min(bandwidth[maxIndex], wt)
                    fringeHeap.Delete(heapIndex[w])
                    heapIndex[w] = fringeHeap.Insert((w,bandwidth[w]))

    return bandwidth, parent

def MBWKruskal(G, s, t):
    edgeList = []
    tree = np.array([i for i in range(G.numVertices)])
    bandwidth = np.zeros(G.numVertices)
    for i in range(G.numVertices):
        for w,wt in G.E[i]:
            if(w>i):
                edgeList.append((i,w,wt))

    edgeList = HeapSort(edgeList,weightIndex=2)