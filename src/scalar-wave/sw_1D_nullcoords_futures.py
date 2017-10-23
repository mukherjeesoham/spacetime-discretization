import numpy as np
import concurrent.futures
import futures_utilities as util
import matplotlib.pyplot as plt

# define the max number of threads the program is allowed to use.
executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

def init(func):						# returns initial/boundary
	return func(dictionary["chebnodes"])

def solve(bcol, brow):          	# returns Patch
	N = dictionary["size"]
	A = dictionary["operator"]
	b = util.makeboundaryvec(N, bcol, brow)
	patch = np.reshape(np.linalg.solve(A, b), (N+1, N+1))
	return patch

def extract(patch, col):			# returns Boundary
	if col == 1:					# extract last column
		return patch[:,  -1]
	else:
		return patch[-1,  :]		# extract last row

def finit(func):                	# returns Future[Boundary]
    return executor.submit(init, func)

def fsetboundary(func):             # returns Future[Boundary]
    return executor.submit(setboundary, func)

def fsolve(fbcol, fbrow):       	# returns Future[Patch]
    return executor.submit(solve, fbcol.result(), fbrow.result())

def fextract(fpatch, col):          # return Future[Boundary]
	if col == 1:
		return executor.submit(extract, fpatch.result(), 1)
	else:
		return executor.submit(extract, fpatch.result(), 0)

def main(M):
	domain = np.zeros((M, M), dtype=object)
	grid   = util.makeglobalgrid(M)	
	for i in range(int(np.max(grid))+1):
		slice = np.transpose(np.where(grid==i))
		for index in slice:
			if np.sum(index) == 0:	# initial patch
				bcol  = finit(lambda x: np.sin(np.pi*x))
				brow  = finit(lambda x: np.sin(np.pi*x))

			elif (np.prod(index) == 0 and np.sum(index) != 0):	
				if index[0] > index[1]:									
					bcol  = finit(lambda x: np.ones(len(x))*np.exp(-10))
					brow  = fextract(domain[index[0] - 1,index[1]], 0)	
				else:													
					brow  = finit(lambda x: np.ones(len(x))*np.exp(-10))
					bcol  = fextract(domain[index[0],index[1]-1], 1)			
			else:												
				bcol  = fextract(domain[index[0],index[1]-1], 1)
				brow  = fextract(domain[index[0]-1,index[1]], 0)
			domain[index[0],index[1]] = fsolve(bcol, brow)
	return domain 	

def assemblegrid(M, domain):
	# FIXME: This is ugly. Fix this!
	I = []
	domain[M-1, M-1].result()
	for i in range(M):
		J = []
		for j in range(M):	
			J.append(domain[i,j].result())
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
# Function calls.
#==================================================================

N = 40	# resolution in a patch
M = 10	# number of patches

dictionary = {
	"size"      : N,
	"chebnodes" : util.cheb(N)[1],
	"operator"  : util.operator(N)
}

domain = assemblegrid(M, main(M))
plotgrid(domain, M, N)