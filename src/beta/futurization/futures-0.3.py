import numpy as np
import concurrent.futures
import utilities as util
import matplotlib.pyplot as plt

# define the max number of threads the program is allowed to use.
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

#==================================================================
# Core functions
#==================================================================

def init(func):					# returns Boundary
	bnd = func(dictionary["chebnodes"])
	return bnd

def solve(bnd1, bnd2):          # returns Patch
	N = dictionary["size"]
	A = dictionary["operator"]
	b = util.makeboundaryvec(N, bnd1, bnd2)
	patch = np.reshape(np.linalg.solve(A, b), (N+1, N+1))
	return patch

def extract(patch, int):		# returns Boundary
	if int == 0:
		return patch[:,  0]
	else:
		return patch[0,  :]

def output(patch):              # returns None
    return None

#==================================================================
# Futurized implementation of the core functions
#==================================================================

def finit(func):                # returns Future[Boundary]
    return executor.submit(init, func)

def fsolve(fbnd1, fbnd2):       # returns Future[Patch]
    return executor.submit(solve, fbnd1.result(), fbnd2.result())

def fextract(fpatch, int):		# return Future[Boundary]
	if int == 0:
		return executor.submit(extract, fpatch.result(), 0)
	else:
		return executor.submit(extract, fpatch.result(), 1)

def foutput(fpatch):            # returns None
    # We don't return a future since we want the output to occur
    # serialized, in a particular order, since all output presumably
    # goes into the same file. If we were to use a more sophisticated
    # output method, then this should also return a future, namely
    # Future[None].
    return output(fpatch.result())

#==================================================================
# Function calls.
#==================================================================

# XXX: Futurize this!
# make a global dictionary all functions can access
N = 2
dictionary = {					
	"size"      : N,
	"chebnodes" : util.cheb(N)[1],
	"operator"  : util.operator(N)
}

"""
To solve for the first patch, we follow
init < solve < output

To solve for subsequent patches
extract < solve < output

For a 4-patch system

: P1 (initial)
	fbnd1   = finit(util.makeinitialdata)
	fbnd2   = finit(util.makeinitialdata)
	fpatch1 = fsolve(fbnd1, fbnd2)

: P2 (boundary)
	fbnd1   = finit(util.makeinitialdata)
	fbnd2   = fextract(fpatch1, 0)				# XXX: How to use fextract correctly? 
	fpatch2 = fsolve(fbnd1, fbnd2)

: P3 (boundary)
	fbnd1   = fextract(fpatch1, 1)
	fbnd2   = finit(util.makeinitialdata)
	fpatch3 = fsolve(fbnd1, fbnd2)

: P4 (boundary)
	fbnd1   = fextract(fpatch2, 1)
	fbnd2   = fextract(fpatch3, 1)
	fpatch3 = fsolve(fbnd1, fbnd2)

"""

maps = util.makeglobalgrid(N)
print maps

for index, element in np.ndenumerate(maps):
	if np.sum(index) == 0:						# first patch
		fbnd1  = finit(util.makeinitialdata)
		fbnd2  = finit(util.makeinitialdata)
		fpatch = fsolve(fbnd1, fbnd2)
	elif np.prod(index) == 0:					# boundary patches
		if index[0] > index[1]:
			print index, element
			fbnd1   = finit(util.makeinitialdata)
			fbnd2   = fextract(fpatch, 0)				
			fpatch  = fsolve(fbnd1, fbnd2)		# doesn't this ruin fpatch for the next element?
		else:
			print index, element
			fbnd1   = fextract(fpatch, 1) 
			fbnd2   = finit(util.makeinitialdata)				
			fpatch  = fsolve(fbnd1, fbnd2)
	else:										# general patch
		print index, element
		fbnd1  = fextract(fpatch, 0)
		fbnd2  = fextract(fpatch, 1)
		fpatch = fsolve(fbnd1, fbnd2)
