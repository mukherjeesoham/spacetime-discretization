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
	domain = [[None]*M]*M
	grid = util.makeglobalgrid(M)
	for i in range(int(np.max(grid))+1):
		slice = np.transpose(np.where(grid==i))
		for index in slice:
			i = index[0]
			j = index[1]
			if np.sum(index) == 0:	# initial patch
				print 60*'-'
				print "Setting future for patch in I", index
				print 60*'-'
				bcol  = finit(lambda x: 0*np.sin(np.pi*x))
				brow  = finit(lambda x: 0*np.sin(np.pi*x))
				domain[i][j] = fsolve(bcol, brow)	

			elif (np.prod(index) == 0 and np.sum(index) != 0):	
				print 60*'-'	
				print "Setting future for patch in B", index
				print 60*'-'	
				if index[0] > index[1]:									
					bcol  = finit(lambda x: np.sin(np.pi*x))
					brow  = fextract(domain[i-1][j], 0)	
					domain[i][j] = fsolve(bcol, brow)
				else:													
					brow  = finit(lambda x: np.sin(np.pi*x))
					bcol  = fextract(domain[i][j-1], 1)
					domain[i][j] = fsolve(bcol, brow)
			
			else:	
				print 60*'-'
				print "Setting future for patch in C", index	
				print 60*'-'													
				bcol  = fextract(domain[i][j-1], 1)
				brow  = fextract(domain[i-1][j], 0)
				domain[i][j] = fsolve(bcol, brow)
			
			print "bcol", bcol.result()
			print "brow", brow.result()
			
			plt.imshow(np.flipud((domain[i][j]).result()))
			plt.show()
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
M = 2

dictionary = {
	"size"      : N,
	"chebnodes" : util.cheb(N)[1],
	"operator"  : util.operator(N)
}

fdomain = main(M)
fdomain[M-1][M-1].result()	# wait for the final result to be computed
print "Finished computation"
print fdomain


domain = np.block([[fdomain[0][0].result(), fdomain[0][1].result()], [fdomain[1][0].result(), fdomain[1][1].result()]])
plt.imshow(np.flipud(domain))
for i in range(M+1):
	plt.axhline([i*N], color='k')
	plt.axvline([i*N], color='k')	
plt.show()
