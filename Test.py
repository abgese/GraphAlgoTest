from Utils import *
from  time import process_time

H = MaxHeap()
start_time = process_time()
for i in range(2500000):
    H.Insert(HeapNode(key=i,weight=i))
print(process_time() - start_time)

start_time = process_time()
while(H.numElements > 0):
    H.Delete(0)
print(process_time() - start_time)