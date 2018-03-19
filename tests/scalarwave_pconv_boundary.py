import numpy as np
import sys
sys.path.append('../')

from scalarwave_multipatch import multipatch
from scalarwave_spectral import spec

def computeL2forpatch(NM):

	#--------------------------------------------------------------
	# Starting to compute solution
	#--------------------------------------------------------------
	if(1):
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
	# Mow plot the solution at a different set of points
	#--------------------------------------------------------------

	"""
	There could be two ways to do this
	- Pad the coefficent matrix with zeros and then evaluate the Vandermonde matrix
	- Compute the Vandermonde matrix in 2D for nmodes != npoints
		- Very confusing with Krons. Doing this explicitly.	
	"""
	
	#--------------------------------------------------------------	
	# Points where you're evaluating the new solution
	#--------------------------------------------------------------
	XNEW, YNEW = spec.chebnodes(computationaldomain.N*2 + 1), \
					spec.chebnodes(computationaldomain.N*2 + 1)
	if(0):
		#--------------------------------------------------------------
		# loop over all new points of the solution [SLOW!] [DEBUG?]
		#--------------------------------------------------------------
		PSOL = np.zeros((computationaldomain.N*2 + 2, computationaldomain.N*2 + 2))
		for i, _x in enumerate(XNEW):		# all points in x-dir
			for j, _y in enumerate(YNEW):	# all points in y-dir			
				SUM = 0
				T   = np.eye(100)
				for index, value in np.ndenumerate(COEFFS):	# sum over all coefficents
					k  = index[0]
					l  = index[1]
					Tk_x = np.polynomial.chebyshev.chebval(_x, T[k])
					Tl_y = np.polynomial.chebyshev.chebval(_y, T[l])
					SUM = SUM + COEFFS[index]*Tk_x*Tl_y
				PSOL[i,j] = SUM
	else:
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
	# Now compute the analytic solution at the same points
	#--------------------------------------------------------------
	ASOL = np.zeros((computationaldomain.N*2 + 2, computationaldomain.N*2 + 2))

	for i, _x in enumerate(XNEW):
		for j, _y in enumerate(YNEW):
			ASOL[i,j] = AF(_x,_y)
	
	#--------------------------------------------------------------
	# Compute L2 norm error
	#--------------------------------------------------------------
	AS = np.ravel(ASOL) 	
	PS = np.ravel(PSOL)
	GW = np.ravel(np.outer(spec.chebweights(computationaldomain.N*2 + 1), \
							spec.chebweights(computationaldomain.N*2 + 1)))
	L2 = np.sqrt(np.abs(np.dot(GW, (PS-AS)**2.0)/np.abs(np.dot(GW, AS**2.0))))

	#--------------------------------------------------------------
	# Plot the solution if necessary
	#--------------------------------------------------------------
	if(0):
		import matplotlib.pyplot as plt
		from mpl_toolkits.mplot3d import Axes3D
		fig = plt.figure()
		ax  = fig.add_subplot(111, projection='3d')

		# xx, yy = np.meshgrid(spec.chebnodes(computationaldomain.N), \
		# 						spec.chebnodes(computationaldomain.N))
		# ax.plot_wireframe(xx, yy, domain, linewidth=0.8)
		
		XX, YY = np.meshgrid(spec.chebnodes(computationaldomain.N*2+1), \
								spec.chebnodes(computationaldomain.N*2+1))
		ax.plot_wireframe(XX, YY, PSOL, color='k', linewidth=0.8)
		ax.plot_wireframe(XX, YY, ASOL, color='g', linewidth=0.8)		
		plt.show()	
	return L2

if(0):
	print "- L2 error norm: %r"%computeL2forpatch(15)
else:
	for i in range(2, 20):
		print "N = %r \t L2: %r" %(i, computeL2forpatch(i))