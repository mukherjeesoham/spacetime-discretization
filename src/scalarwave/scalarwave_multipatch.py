#--------------------------------------------------------------------
# Spacetime Discretization methods Scalar Wave Prototype 
# Utilities for spectral integration and differentiation
# Soham M 10-2017
#--------------------------------------------------------------------
import numpy as np
from scipy.integrate import quad, dblquad
from scalarwave_spectral import spec
from scalarwave_patch import patch

class multipatch(object):
	"""
	The main function, which after futurization,
	takes the size and number of patches, the boundary conditions/the potential
	and computes, and then assembles the entire solution
	"""
	def __init__(self, npatches, nmodes, leftboundary, rightboundary, potential):
		self.M      = npatches
		self.N 		= nmodes
		self.funCOL = leftboundary
		self.funROW = rightboundary
		self.funcV  = potential

	@staticmethod
	def makeglobalgrid(M):
		grid = np.zeros((M,M))
		for index, val in np.ndenumerate(grid):
			grid[index] = np.sum(index)
		return grid

	@staticmethod
	def shifts(M):
		L = np.zeros((M, M), dtype=object)
		for k, i in enumerate(range(-M+1, M+1, 2)):
			for l, j in enumerate(range(-M+1, M+1, 2)):
				L[k, l] =  np.array([i, -j])
		return L.T

	def computelocalV(self, XP, YP):
		XX, YY = np.meshgrid(XP, YP)
		if self.funcV == None:
			return (XX + YY)*0
		else:
			return self.funcV(XX, YY)

	def gridtopatch(self, PATCH, index):
		"""
		Gives the integration weights and the 
		"""
		M, N    = self.M, self.N
		S, WS 	= self.shifts(M), spec.chebweights(N)
		X, Y  	= spec.chebnodes(PATCH.N), spec.chebnodes(PATCH.N)	
 		CC 		= np.insert(np.cumsum(WS) - 1, 0, -1)

		self.PX = np.sort((X + S[index[0], index[1]][0])/M)
		self.PY = np.sort(-(Y + S[index[0], index[1]][1])/M)
		self.WX = np.sort((CC + S[index[0], index[1]][0])/M)
		self.WY = np.sort(-(CC + S[index[0], index[1]][1])/M)

		return self.WX, self.WY, self.PX, self.PY

	def globalsolve(self):
		domain   = np.zeros((self.M, self.M), dtype=object)
		grid     = self.makeglobalgrid(self.M)	
		PATCH    = patch(self.N)
		OPERATOR = patch.operator(PATCH)

		for i in range(int(np.max(grid))+1):
			slice = np.transpose(np.where(grid==i))
			for index in slice:
				PATCH.loc = index
				if not self.M == 1:
					print "Computing patch: ", index
				CX, CY, XP, YP = self.gridtopatch(PATCH, index) 	

				if np.sum(index) == 0:	
					BC = patch.setBCs(PATCH, BROW = spec.projectfunction1D(self.funROW, 50, XP), \
											 BCOL = spec.projectfunction1D(self.funCOL, 50, YP), \
											 PV = self.computelocalV(XP, YP))	
				elif (np.prod(index) == 0 and np.sum(index) != 0):	
					if index[1] > index[0]:	
						BC = patch.setBCs(PATCH, BROW = spec.projectfunction1D(self.funROW, 50, XP), \
							 					 BCOL = patch.extractpatchBC(domain[index[0], index[1]-1], index), \
							 					 PV = self.computelocalV(XP, YP))						
					else:
						BC = patch.setBCs(PATCH, BROW = patch.extractpatchBC(domain[index[0]-1,  index[1]], index), \
				 					 			 BCOL = spec.projectfunction1D(self.funCOL, 50, YP), \
							 					 PV = self.computelocalV(XP, YP))	
				else:		
					BC = patch.setBCs(PATCH, BROW = patch.extractpatchBC(domain[index[0]-1,  index[1]], index), \
			 			 					 BCOL = patch.extractpatchBC(domain[index[0], index[1]-1], index), \
 					 						 PV = self.computelocalV(XP, YP))									
				
				# Call solver: returns a patch object
				domain[index[0], index[1]] = PATCH.solve(BC, OPERATOR).patchval					

				# Plot the patch # issues with plotting the patch
				patch.plotpatch(domain[index[0], index[1]], CX, CY, XP, YP)

		return domain
