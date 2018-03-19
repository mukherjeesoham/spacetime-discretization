import numpy as np
import sys
sys.path.append('../')

from scalarwave_multipatch import multipatch
from scalarwave_spectral import spec

def computeL2forpatch(NM):

	sinwave = 0

	#--------------------------------------------------------------
	# Starting to compute solution
	#--------------------------------------------------------------
	if(sinwave):
		# Two gaussians advecting
		computationaldomain = multipatch(npatches=1, nmodes=20, \
								leftboundary   = lambda x: np.exp(-(x**2.0)/0.1), \
								rightboundary  = lambda y: np.exp(-(y**2.0)/0.1), \
								potential 	   = None,
								nboundarymodes = NM)
		AF	 = lambda x, y: np.exp(-(x**2.0)/0.1) + np.exp(-(y**2.0)/0.1)
	else:
		# Two sine waves advecting
		computationaldomain = multipatch(npatches=1, nmodes=NM, \
								leftboundary   = lambda x: np.sin(np.pi*x), \
								rightboundary  = lambda y: np.sin(np.pi*y), \
								potential 	   = None,
								nboundarymodes = NM)
		AF	 = lambda x, y: -np.sin(np.pi*x) + -np.sin(np.pi*y)

	#--------------------------------------------------------------
	# Call solver
	#--------------------------------------------------------------
	domain = computationaldomain.globalsolve()[0,0]

	#--------------------------------------------------------------
	# Extract coefficents from the patch
	#--------------------------------------------------------------
	CP 	   = spec.vandermonde(computationaldomain.N, \
								spec.chebnodes(computationaldomain.N))
	VNDM   = np.kron(CP, CP)  
	CFS    = np.linalg.solve(VNDM, np.ravel(domain))
	COEFFS = np.reshape(CFS, (computationaldomain.N+1, computationaldomain.N+1))
	
	#--------------------------------------------------------------	
	# Points where you're evaluating the new solution
	#--------------------------------------------------------------
	XNEW, YNEW = spec.chebnodes(computationaldomain.N*2 + 1), \
					spec.chebnodes(computationaldomain.N*2 + 1)

	#--------------------------------------------------------------
	# Pad coefficent matrix with zeros
	#--------------------------------------------------------------
	COEFFS_P = np.pad(COEFFS, \
					(0, (computationaldomain.N*2 + 1) - computationaldomain.N), 'constant')
	CQP= spec.vandermonde(computationaldomain.N*2 + 1, \
						spec.chebnodes(computationaldomain.N*2 + 1))
	VNDM_P   = np.kron(CQP, CQP) 
	PSOLV    = np.dot(VNDM_P, np.ravel(COEFFS_P))
	PSOL     = np.reshape(PSOLV, (computationaldomain.N*2 + 2, computationaldomain.N*2 + 2))

	#--------------------------------------------------------------
	# Now compute the lower mode boundary solution solution
	#--------------------------------------------------------------
	
	if(sinwave):
		# Two gaussians advecting
		computationaldomain = multipatch(npatches=1, nmodes=20, \
								leftboundary   = lambda x: np.exp(-(x**2.0)/0.1), \
								rightboundary  = lambda y: np.exp(-(y**2.0)/0.1), \
								potential 	   = None,
								nboundarymodes = NM+1)
		AF	 = lambda x, y: np.exp(-(x**2.0)/0.1) + np.exp(-(y**2.0)/0.1)
	else:
		# Two sine waves advecting
		computationaldomain = multipatch(npatches=1, nmodes=NM, \
								leftboundary   = lambda x: np.sin(np.pi*x), \
								rightboundary  = lambda y: np.sin(np.pi*y), \
								potential 	   = None,
								nboundarymodes = NM+1)
		AF	 = lambda x, y: -np.sin(np.pi*x) + -np.sin(np.pi*y)

	#--------------------------------------------------------------
	# Call solver
	#--------------------------------------------------------------
	domain = computationaldomain.globalsolve()[0,0]

	#--------------------------------------------------------------
	# Extract coefficents from the patch
	#--------------------------------------------------------------
	CP 	   = spec.vandermonde(computationaldomain.N, \
								spec.chebnodes(computationaldomain.N))
	VNDM   = np.kron(CP, CP)  
	CFS    = np.linalg.solve(VNDM, np.ravel(domain))
	COEFFS = np.reshape(CFS, (computationaldomain.N+1, computationaldomain.N+1))

	#--------------------------------------------------------------
	# Now, plot the solution at a different set of points
	#--------------------------------------------------------------

	XNEW, YNEW = spec.chebnodes(computationaldomain.N*2 + 1), \
					spec.chebnodes(computationaldomain.N*2 + 1)

	#--------------------------------------------------------------
	# Pad coefficent matrix with zeros
	#--------------------------------------------------------------
	COEFFS_P = np.pad(COEFFS, \
					(0, (computationaldomain.N*2 + 1) - computationaldomain.N), 'constant')
	CQP= spec.vandermonde(computationaldomain.N*2 + 1, \
						spec.chebnodes(computationaldomain.N*2 + 1))
	VNDM_P   = np.kron(CQP, CQP) 
	PSOLV    = np.dot(VNDM_P, np.ravel(COEFFS_P))
	ASOL     = np.reshape(PSOLV, (computationaldomain.N*2 + 2, computationaldomain.N*2 + 2))

	
	#--------------------------------------------------------------
	# Compute L2 norm error
	#--------------------------------------------------------------
	AS = np.ravel(ASOL) 	
	PS = np.ravel(PSOL)
	GW = np.ravel(np.outer(spec.chebweights(computationaldomain.N*2 + 1), \
							spec.chebweights(computationaldomain.N*2 + 1)))
	L2 = np.sqrt(np.abs(np.dot(GW, (PS-AS)**2.0)/np.abs(np.dot(GW, AS**2.0))))
	return L2

if(0):
	print "- L2 error norm: %r"%computeL2forpatch(15)
else:
	for i in range(2, 20):
		print "N = %r \t L2: %r" %(i, computeL2forpatch(i))