import numpy as np
import concurrent.futures
import futures_utilities as util
import matplotlib.pyplot as plt

# define the max number of threads the program is allowed to use.
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

#==================================================================
# Core functions
#==================================================================

def init(func):					# returns initial/boundary
	return func(dictionary["chebnodes"])

def solve(bcol, brow):          # returns Patch
	N = dictionary["size"]
	A = dictionary["operator"]
	b = util.makeboundaryvec(N, bcol, brow)
	patch = np.reshape(np.linalg.solve(A, b), (N+1, N+1))
	return patch

def extract(patch, col):		# returns Boundary
	if col == 1:				# extract last column
		return patch[:,  -1]
	else:
		return patch[-1,  :]	# extract last row

def output(patch):          
    return None

#==================================================================
# Futurized implementation of the core functions
#==================================================================

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
	domain = {}
	grid = util.makeglobalgrid(M)
	for i in range(int(np.max(grid))+1):
		slice = np.transpose(np.where(grid==i))
		for index in slice:
			i = index[0]
			j = index[1]
			if np.sum(index) == 0:	# initial patch
				bcol  = init(lambda x: 0*np.sin(np.pi*x))
				brow  = init(lambda x: 0*np.sin(np.pi*x))

			elif (np.prod(index) == 0 and np.sum(index) != 0):	
				if index[0] > index[1]:									
					bcol  = init(lambda x: np.sin(np.pi*x))
					brow  = extract(domain[str(i-i)+str(j)], 0)	
				else:													
					brow  = init(lambda x: np.sin(np.pi*x))
					bcol  = extract(domain[str(i)+str(j-1)], 1)			
			else:														
				bcol  = extract(domain[str(i)+str(j-1)], 1)
				brow  = extract(domain[str(i-1)+str(j)], 0)
			
			domain[str(i)+str(j)] = solve(bcol, brow)
	return domain 	

def assemblegrid(M, domain):
	I = []
	for i in range(M):
		J = []
		for j in range(M):	
			J.append(domain[str(i)+str(j)])
		I.append(J)
	return np.block(I)

#==================================================================
# Function calls.
#==================================================================

N = 40
M = 10

dictionary = {
	"size"      : N,
	"chebnodes" : util.cheb(N)[1],
	"operator"  : util.operator(N)
}

domain = assemblegrid(M, main(M))
print np.shape(domain)
plt.imshow(domain)
for k in range(M+1):
	plt.axvline([k*N], color='w')
	plt.axhline([k*N], color='w')


plt.show()
