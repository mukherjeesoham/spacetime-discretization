import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
import math

# define the max number of threads the program is allowed to use.
executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

def cheb(N):
	x  = np.cos(math.pi*np.arange(0,N+1)/N)
	c  = np.concatenate(([2],np.ones(N-1),[2]))*(-1)**np.arange(0,N+1)
	X  = np.tile(x,(N+1,1))
	dX = X-np.transpose(X)
	c  = np.transpose(c)
	D  = -np.reshape(np.kron(c,1/c),(N+1,N+1))/(dX+np.eye(N+1))
	D  = D - np.diagflat(np.sum(D,axis=1))	# diagonal entries

	return D,x

def clencurt(N):
	"""
	see <https://github.com/mikaem/spmpython>
	CLENCURT nodes x (Chebyshev points) and weights w 
	for Clenshaw-Curtis quadrature
	"""
	theta = np.pi*np.arange(0,N+1)/N
	x  = np.cos(theta)
	w  = np.zeros(N+1)
	ii = np.arange(1,N)
	v  = np.ones(N-1)
	if np.mod(N,2)==0:
		w[0] = 1./(N**2-1)
		w[N] = w[0]
		for k in np.arange(1,int(N/2.)):
		    v = v-2*np.cos(2*k*theta[ii])/(4*k**2-1)
		v = v - np.cos(N*theta[ii])/(N**2-1)
	else:
		w[0] = 1./N**2
		w[N] = w[0]
	for k in np.arange(1,int((N-1)/2.)+1):
	    v = v-2*np.cos(2*k*theta[ii])/(4*k**2-1)
	w[ii] = 2.0*v/N
	return w

def makeboundaryvec(N, bcol, brow):
	b = np.eye(N+1)*0.0
	b[:,  0] = bcol
	b[0,  :] = brow
	return np.ravel(b)

def makeglobalgrid(M):
	grid = np.zeros((M,M))
	for index, val in np.ndenumerate(grid):
		grid[index] = np.sum(index)
	return grid

def operator(N):
	DU, U  = cheb(N)	# conformally compactified metric coordinates
	DV, V  = cheb(N)

	I  = np.eye(N+1)
	DU = np.kron(DU, I)
	DV = np.kron(I, DV)
	
	detJ     = np.sqrt(2.0)
	OmegaSq  = (U**2.0 - 1.0)*(V**2.0 - 1.0)
	sqrtdetG = 1.0/(OmegaSq + 1.0)	# blows up
	lnOmega  = np.log(OmegaSq)/2.0
	
	# Matter + Curvature Lagrangians
	LM = np.dot(DU,DV) + np.dot(DV,DU)
	LG = (np.dot(DU, sqrtdetG*np.dot(DU, lnOmega)) + 
				np.dot(DV, sqrtdetG*np.dot(DV, lnOmega)))

	# D = (LM + LG)*detJ

	# # integration weights
	# V  = np.outer(clencurt(N), clencurt(N))
	# W  = np.diag(np.ravel(V))                
	# A  = W.dot(D)
	# BC = np.zeros((N+1,N+1))
	# BC[0, :] = BC[:, 0]  = 1  
	# A[np.where(np.ravel(BC)==1)[0]] = np.eye((N+1)**2)[np.where(np.ravel(BC)==1)[0]]  
	return None

operator(4)

def init(func):						# returns initial/boundary
	return func(dictionary["chebnodes"])

def solve(bcol, brow):          	# returns Patch
	N = dictionary["size"]
	A = dictionary["operator"]
	b = makeboundaryvec(N, bcol, brow)
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
	grid   = makeglobalgrid(M)	
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
	# plt.axis("off")
	plt.show()

#==================================================================
# Function calls.
#==================================================================

# if{0}:
# 	N = 40	# resolution in a patch
# 	M = 1	# number of patches

# 	dictionary = {
# 		"size"      : N,
# 		"chebnodes" : cheb(N)[1],
# 		"operator"  : operator(N)
# 	}

# 	domain = assemblegrid(M, main(M))
# 	plotgrid(domain, M, N)

