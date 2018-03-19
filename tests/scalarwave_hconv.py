import numpy as np
import sys
sys.path.append('../')

from scalarwave_multipatch import multipatch
from scalarwave_spectral import spec

global nmodesP
nmodesP = 4

def shifts(M):
	L = np.zeros((M, M), dtype=object)
	for k, i in enumerate(range(-M+1, M+1, 2)):
		for l, j in enumerate(range(-M+1, M+1, 2)):
			L[k, l] =  np.array([i, -j])
	return L.T

def gridtopatch(NM, index):
	M, N    = NM, nmodesP
	S, WS 	= shifts(M), spec.chebweights(N)
	X, Y  	= spec.chebnodes(N), spec.chebnodes(N)	
	CC 		= np.insert(np.cumsum(WS) - 1, 0, -1)

	PX = np.sort((X + S[index[0], index[1]][0])/M)
	PY = np.sort(-(Y + S[index[0], index[1]][1])/M)

	return PX, PY

def computeL2forgrid(NM):

	#--------------------------------------------------------------
	# Starting to compute solution
	#--------------------------------------------------------------
	if(0):
		computationaldomain = multipatch(npatches=NM, nmodes=nmodesP, \
								leftboundary  = lambda x: np.exp(-(x**2.0)/0.1), \
								rightboundary = lambda y: np.exp(-(y**2.0)/0.1), \
								potential 	  = None)
		AF	 = lambda x, y: np.exp(-(x**2.0)/0.1) + np.exp(-(y**2.0)/0.1)
	else:
		computationaldomain = multipatch(npatches=NM, nmodes=nmodesP, \
								leftboundary  = lambda x: np.sin(np.pi*x), \
								rightboundary = lambda y: np.sin(np.pi*y), \
								potential 	  = None)
		AF	 = lambda x, y: np.sin(np.pi*x) + np.sin(np.pi*y)

	#--------------------------------------------------------------
	# Call solver
	#--------------------------------------------------------------
	grid = computationaldomain.globalsolve()

	#--------------------------------------------------------------
	# Project the solution on each patch
	#--------------------------------------------------------------
	for index_grid, val in np.ndenumerate(grid):
		
		domain = grid[index_grid]
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
		X, Y       = spec.chebnodes(computationaldomain.N), spec.chebnodes(computationaldomain.N)
		XNEW, YNEW = gridtopatch(NM, index_grid)

		#--------------------------------------------------------------
		# loop over all new points of the solution [SLOW!] [DEBUG]
		#--------------------------------------------------------------
		PSOL = np.zeros((len(XNEW), len(YNEW)))
		for i, _x in enumerate(XNEW):		# all points in x-dir
			for j, _y in enumerate(YNEW):	# all points in y-dir			
				SUM = 0
				T   = np.eye(100)
				for index, value in np.ndenumerate(COEFFS):	# sum over all coefficents
					k  = index[0]
					l  = index[1]
					Tk_x = np.polynomial.chebyshev.chebval(_x, T[k])
					Tl_y = np.polynomial.chebyshev.chebval(_y, T[l])
					SUM  = SUM + COEFFS[index]*Tk_x*Tl_y
				PSOL[i,j] = SUM
		
		#--------------------------------------------------------------
		# Now compute the analytic solution at the same points
		#--------------------------------------------------------------
		ASOL = np.zeros((len(XNEW), len(YNEW)))
		for i, _x in enumerate(XNEW):
			for j, _y in enumerate(YNEW):
				ASOL[i,j] = AF(_x,_y)
	
		if(1):
			import matplotlib.pyplot as plt
			# from mpl_toolkits.mplot3d import Axes3D
			# fig = plt.figure()
			# ax  = fig.add_subplot(111, projection='3d')
			
			# XX, YY = np.meshgrid(spec.chebnodes(nmodesP), \
			# 						spec.chebnodes(nmodesP))
			# ax.plot_wireframe(XX, YY, PSOL, color='k', linewidth=0.8)
			# ax.plot_wireframe(XX, YY, ASOL, color='g', linewidth=0.8)		

			plt.subplot(131)
			plt.contourf(X, Y, domain)
			plt.colorbar()
			plt.subplot(132)
			plt.contourf(XNEW, YNEW, PSOL)
			plt.colorbar()
			plt.subplot(133)
			plt.contourf(XNEW, YNEW, ASOL)
			plt.colorbar()
			plt.show()	

		np.set_printoptions(precision=4)
		#--------------------------------------------------------------
		# Compute L2 norm error
		#--------------------------------------------------------------
		AS = np.ravel(ASOL) 	
		PS = np.ravel(PSOL)
		# print 50*"-"
		# print index_grid
		# print AS
		# print PS
		# print 50*"-"
		GW = np.ravel(np.outer(spec.chebweights(nmodesP), \
								spec.chebweights(nmodesP)))
		L2 = np.sqrt(np.abs(np.dot(GW, (PS-AS)**2.0)/np.abs(np.dot(GW, AS**2.0))))
		print "PATCH %r L2 = %r"%(index_grid, L2)
	return L2

if(1):
	computeL2forgrid(2)
else:
	for i in range(2, 20):
		print "N = %r \t L2: %r" %(i, computeL2forgrid(i))