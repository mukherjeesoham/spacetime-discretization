import numpy as np
import concurrent.futures
import futures_utilities as util
import matplotlib.pyplot as plt

# define the max number of threads the program is allowed to use.
executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

def init(func):						# returns initial/boundary
	return func(dictionary["chebnodes"])

def solve(b):          	# returns Patch
	N = dictionary["size"]
	A = dictionary["operator"]
	B = np.ravel(b)
	patch = np.reshape(np.linalg.solve(A, B), (N+1, N+1))
	return patch

def extract(patch, col):			# returns Boundary
	if col == 1:					# extract last column
		return patch[:,  -1]
	else:
		return patch[-1,  :]		# extract last row

def main(M):
	domain = np.zeros((M, M), dtype=object)
	grid   = util.makeglobalgrid(M)	
	for i in range(int(np.max(grid))+1):
		slice = np.transpose(np.where(grid==i))
		for index in slice:
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
	return domain 	

def assemblegrid(M, domain):
	I = []
	domain[M-1, M-1]
	for i in range(M):
		J = []
		for j in range(M):	
			J.append(domain[i,j])
		I.append(J)
	
	block = np.block(I)
	columns = np.linspace(0, M*(N+1), M+1)[1:-1]
	block = np.delete(block, columns, 0) 
	block = np.delete(block, columns, 1) 
	return block

def plotgrid(domain, M, N):
	plt.imshow(domain)
	for k in range(1, M):
		plt.axvline([k*(N)], color='w')
		plt.axhline([k*(N)], color='w')
	plt.axis("off")
	plt.show()

#==================================================================
# Function calls
#==================================================================

N = 60	# resolution in a patch
M = 4	# number of patches

dictionary = {
	"size"      : N,
	"chebnodes" : util.cheb(N)[1],
	"operator"  : util.operator(N),
	"potential" : util.makeglobalchart(M,N)
}

domain = assemblegrid(M, main(M))
plotgrid(domain, M, N)