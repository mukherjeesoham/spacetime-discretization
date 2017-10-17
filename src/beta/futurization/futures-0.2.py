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

def finit(func):                 # returns Future[Boundary]
    return executor.submit(init, func)

def fsolve(fbnd1, fbnd2):       # returns Future[Patch]
    return executor.submit(solve, fbnd1.result(), fbnd2.result())

def fextract(fpatch, int):          # return Future[Boundary]
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
dictionary = {					
	"size"      : 10,
	"chebnodes" : util.cheb(10)[1],	# futurize this!
	"operator"  : util.operator(10)	# and this!
}

# To solve for the first patch, we follow
# init < solve < output
# To solve for subsequent patches
# extract < solve < output

# FIXME: How to encode this order?

# Patch 00
fbnd1_00  = finit(util.makeinitialdata)
fbnd2_00  = finit(util.makeinitialdata)
fpatch_00 = fsolve(fbnd1_00, fbnd2_00)

# Patch 10
fbnd1_10    = finit(util.makeinitialdata)
fbnd2_10    = fextract(fpatch_00, 1)
fpatch_10 = fsolve(fbnd1_10, fbnd2_10)

# Patch 01
fbnd1_01    = fextract(fpatch_00, 0)
fbnd2_01    = finit(util.makeinitialdata)
fpatch_01 = fsolve(fbnd1_01, fbnd2_01)
	
# Patch 22	# needs boundary conditions from adjacent
fbnd1_22  = fextract(fpatch_10, 0)
fbnd2_22  = fextract(fpatch_01, 1)
fpatch_22 = fsolve(fbnd1_22, fbnd2_22)