#===============================================================
# Scalar wave equation in Minkowski Spacetime
# Soham M 10/2017
#===============================================================

import numpy as np
import concurrent.futures
import scalarwave_utilities as util
import matplotlib.pyplot as plt

def init(func):						# returns initial/boundary
	return func(dictionary["chebnodes"])

def solve(b):          				# returns Patch
	N = dictionary["size"]
	A = dictionary["operator"]
	return np.reshape(np.linalg.solve(A, np.ravel(b)), (N+1, N+1))

def extract(patch, col):			# returns Boundary
	if col == 1:					# extract last column
		return patch[:,  -1]
	else:
		return patch[-1,  :]		# extract last row

def main(M, N):
	domain = np.zeros((M, M), dtype=object)
	grid   = util.makeglobalgrid(M)	
	print "P: ", dictionary["size"] + 1
	print "H: ", dictionary["numpatches"]
	util.println()
	print "==> Starting computation"
	util.println()
	for i in range(int(np.max(grid))+1):
		slice = np.transpose(np.where(grid==i))
		for index in slice:
			print "Computing patch: ", index
			B = dictionary["potential"][index[1], index[0], :, :]
			if np.sum(index) == 0:	# initial patch
				bcol  = init(lambda x: np.zeros(len(x)))
				brow  = init(lambda x: np.zeros(len(x)))

			elif (np.prod(index) == 0 and np.sum(index) != 0):	
				if index[0] > index[1]:									
					bcol  = init(lambda x: np.zeros(len(x)))
					brow  = extract(domain[index[0] - 1,index[1]], 0)	
				else:													
					brow  = init(lambda x: np.zeros(len(x)))
					bcol  = extract(domain[index[0],index[1]-1], 1)			
			else:												
				bcol  = extract(domain[index[0],index[1]-1], 1)
				brow  = extract(domain[index[0]-1,index[1]], 0)
			
			B[0, :] = brow	# we need these to set BCs at edges
			B[:, 0] = bcol	# BCs at edges
			domain[index[0],index[1]] = solve(B)
	return util.assemblegrid(M, N, domain)

#==================================================================
# Function calls
#==================================================================


#--------------------------------------------------------------------
# test coefficents
#--------------------------------------------------------------------
if(0):
	dictionary["domain"] = main(M, N)
	util.plotgrid(dictionary)
	plt.semilogy(np.abs(util.extractcoeffs(dictionary["domain"])))
	plt.xlabel("$\mu$")
	plt.ylabel("$c_{\mu}$")
	plt.show()

#--------------------------------------------------------------------
# test p-convergence
#--------------------------------------------------------------------

patches = []
for N in range(12,20,1):
	M = 1	# number of patches

	dictionary = {
		"size"       : N,
		"numpatches" : M,
		"chebnodes"  : util.cheb(N)[1],
		"operator"   : util.operator(N)[0],
		"potential"  : util.makeglobalchartcopy(M,N)[1]}
	patches.append(main(M,N))


for index, patch in enumerate(patches):
	print np.shape(util.prolongate(patch, index, index+1))
	
