import numpy as np
import concurrent.futures
import futures_utilities as util
import matplotlib.pyplot as plt

# define the max number of threads the program is allowed to use.
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

#==================================================================
# Core functions
#==================================================================

def init(func):					# returns Initial boundary
	bnd = func(dictionary["chebnodes"])	# this is a very rudimentary implementation
	return bnd

def solve(bnd1, bnd2):          # returns Patch
	N = dictionary["size"]
	A = dictionary["operator"]
	b = util.makeboundaryvec(N, bnd1, bnd2)
	patch = np.reshape(np.linalg.solve(A, b), (N+1, N+1))
	return patch

def extract(patch, int):		# returns Boundary
	if int == 0:				# extract right boundary
		return patch[:,  0]
	else:
		return patch[0,  :]		# extract left boundary

def output(patch):          
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
# Main loop
#==================================================================

def main(M):
	domain = [[None]*M]*M
	grid = util.makeglobalgrid(M)
	for i in range(int(np.max(grid))+1):
		slice = np.transpose(np.where(grid==i))
		for index in slice:
			if np.sum(index) == 0:										# initial patch
				bnd1  = finit(util.makeinitialdata)
				bnd2  = finit(util.makeinitialdata)	
			elif np.prod(index) == 0:									# boundary patches
				if index[0] > index[1]:									
					bnd1  = finit(util.makeinitialdata)
					bnd2  = fextract(domain[index[0]-1][index[1]], 1)	
				else:													
					bnd2  = finit(util.makeinitialdata)
					bnd1  = fextract(domain[index[0]][index[1]-1], 0)
			else:														# some patch in the middle.
				bnd1  = fextract(domain[index[0]][index[1]-1], 0)
				bnd2  = fextract(domain[index[0]-1][index[1]], 1)
			domain[index[0]][index[1]] = fsolve(bnd1, bnd2) 
	return domain 	

def assemblegrid(M, fdomain):
	I = []
	for i in range(M):
		J = []
		for j in range(M):	
			J.append(fdomain[i][j].result())
		I.append(J)
	return np.block(I)

#==================================================================
# Function calls.
#==================================================================

N = 20
M = 4

dictionary = {
	"size"      : N,
	"chebnodes" : util.cheb(N)[1],
	"operator"  : util.operator(N)
}

fdomain = main(M)
fdomain[M-1][M-1].result()	# wait for the final result to be computed
domain = assemblegrid(M, fdomain)
plt.imshow(domain)												
plt.show()
