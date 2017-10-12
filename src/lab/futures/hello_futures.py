#!/usr/bin/env python
 
from Queue import Queue
import random
import time
 
q = Queue()
fred = [1,2,3,4,5,6,7,8,9,10]
 
def f(x):
    if random.randint(0,1):
        print "sleeping"
        time.sleep(0.2)
    #
    res = x * x
    q.put(res)
 
def main():
    for num in fred:
        f(num)
    #
    while not q.empty():
        print q.get()
 
if __name__ == "__main__":
    main()
