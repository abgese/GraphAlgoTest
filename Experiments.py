from Utils import *
from time import process_time
import numpy as np


AVG_NUM_EDGES = 6
EDGE_PROB = 0.2 # Can be interpreted as being closed to % of edges in graph picked
NUM_VERTICES = 5000
MAX_EDGE_WEIGHT = 1000



Results = {
    "DjikstraNoHeap" : {"Type_G1" : [], "Type_G2" : []}, 
    "DjikstraWithHeap" : {"Type_G1" : [], "Type_G2" : []}, 
    "Kruskals" : {"Type_G1" : [], "Type_G2" : []}
    }

Tests = []

for i in range(5):
    G1 = Graph(numVertices = NUM_VERTICES, maxEdgeWeight=MAX_EDGE_WEIGHT, avgNumEdges=AVG_NUM_EDGES)
    G2 = Graph(numVertices = NUM_VERTICES, maxEdgeWeight=MAX_EDGE_WEIGHT, edgeProb=EDGE_PROB)
    Tests.append(("\nExperiment : {}".format(i + 1),""))
    for j in range(5):
        s = np.random.randint(0,NUM_VERTICES)
        t = s
        while( t == s):
            #Ensure s != t, However unlikely
            t = np.random.randint(0,NUM_VERTICES)

        Tests.append((s,t))

        #Djikstra Without heap
        start_time = process_time() # return time taken by process specifically , based on CPU cycles. Better measure than time()
        bandwidth, parent = MBWDjikstraNoHeap(G1, s, t)
        elapsed_time = process_time() - start_time
        Results["DjikstraNoHeap"]["Type_G1"].append(elapsed_time)

        start_time = process_time()
        bandwidth, parent = MBWDjikstraNoHeap(G2, s, t)
        elapsed_time = process_time() - start_time
        Results["DjikstraNoHeap"]["Type_G2"].append(elapsed_time)

        #Djikstra With heap
        start_time = process_time()
        bandwidth, parent = MBWDjikstraHeap(G1, s, t)
        elapsed_time = process_time() - start_time
        Results["DjikstraWithHeap"]["Type_G1"].append(elapsed_time)

        start_time = process_time()
        bandwidth, parent = MBWDjikstraHeap(G2, s, t)
        elapsed_time = process_time() - start_time
        Results["DjikstraWithHeap"]["Type_G2"].append(elapsed_time)

        #Kruskals
        start_time = process_time() 
        bandwidth, parent = MBWKruskal(G1, s, t)
        elapsed_time = process_time() - start_time
        Results["Kruskals"]["Type_G1"].append(elapsed_time)

        start_time = process_time()
        bandwidth, parent = MBWKruskal(G2, s, t)
        elapsed_time = process_time() - start_time
        Results["Kruskals"]["Type_G2"].append(elapsed_time)
    

with open('Results.txt', 'w+') as f:
    for k in Results.keys():
        f.write("\n" + k + "\n")
        f.write("G1\tG2\n")
        for i in range(len(Results[k]["Type_G1"])):
            f.write("{}\t{}\n".format(Results[k]["Type_G1"][i], Results[k]["Type_G2"][i]))

with open('Tests.txt', 'w+') as f:
    for k in Tests:
        f.write('{}\t{}\n'.format(k[0],k[1]))
