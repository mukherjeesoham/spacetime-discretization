#--------------------------------------------------------------------
# Spacetime Discretization methods Scalar Wave Prototype 
# Utilities hanlding patch handling and computation
# Soham M 10-2017
#--------------------------------------------------------------------

import numpy as np
from scipy.integrate import quad, dblquad
from scalarwave_spectral import spec

class patch(spec):
	def __init__(self, N, loc = None):
		self.N   	  = N
		self.loc 	  = loc
		self.operator = None
		self.bcmatrix = None

	def integrationweights(self):
		# returns a 2D diagonal matrix
		N = self.N
		return np.diag(np.ravel(np.outer(spec.chebweights(N), \
			spec.chebweights(N))))

	# TODO: Remove this explicit construction and allow for a higher level
	# description of the operator
	def operator(self):
		# TODO: Optimize this computation. Can we construct OP element-wise <use Julia>?
		N = self.N
		DU = np.kron(spec.chebmatrix(N), np.eye(N+1))
		DV = np.kron(np.eye(N+1), spec.chebmatrix(N))
		D  = np.dot(DU,DV) + np.dot(DV,DU)

		# NOTE: We do not symmetrize the operator!
		OPERATOR = self.integrationweights().dot(D)	
		
		# Replace rows in the matrix to implement Dirichlet boundary conditions 
		BC = np.zeros((N+1,N+1))
		BC[0, :] = BC[:, 0] = 1 # Set Dirichlet BCs at adjacent edges
		OPERATOR[np.where(np.ravel(BC)==1)[0]] = np.eye((N+1)**2)[np.where(np.ravel(BC)==1)[0]]  
		self.operator = OPERATOR	

		return self.operator

	def eigenval(self):
		print "==> Computing eigenvalues"
		eigenvalues = np.linalg.eigvals(self.operator)
		emax = np.amax(np.abs(eigenvalues))
		emin = np.amin(np.abs(eigenvalues))
		print "   - Eigenval (max/min): ", emax/emin
		return eigenvalues
  
	@staticmethod
	def extractpatchBC(PATCHVAL, column):		
		if column == 1:			
			return PATCHVAL[:,  -1]
		else:	
			return PATCHVAL[-1,  :]	

	# FIXME: Clean this up.
	def setBCs(self, BROW, BCOL, PV = None):	
		"""
		Computes the boundary condition + potential array. 
		Note that none of the inputs are necessary to call solve.
		"""
		if not isinstance(PV, np.ndarray):
			PBC = np.zeros((self.N + 1, self.N + 1))
		else:
			PBC = PV
			
		# NOTE: Multiply the potential with the integration weights since it 
		# appears under the integral sign in the action
		PBC = np.reshape(np.multiply(np.diag(self.integrationweights()), \
							np.ravel(PBC)), (self.N+1, self.N+1))

		PBC[0, :] = BROW
		PBC[:, 0] = BCOL

		self.bcmatrix = PBC
		return self.bcmatrix

	@staticmethod
	def computelocalV(funcV, XP, YP):
		XX, YY = np.meshgrid(XP, YP)
		if funcV == None:
			return (XX + YY)*0
		else:
			return funcV(XX, YY)

	def solve(self, boundaryconditions, operator):	
		self.patchval = np.reshape(np.linalg.solve(operator, \
			np.ravel(boundaryconditions)), (self.N+1, self.N+1))
		return self

	@staticmethod
	def extractpatchcoeffs(self):
		CP = spec.vandermonde1D(self.N, self.N)
		CM = np.kron(CP, CP)  
		return np.reshape(np.linalg.solve(CM, np.ravel(self.patchval)), \
			(self.N+1, self.N+1))

	def computepatchvalues(self, coefficents):
		CP = spec.vandermonde1D(self.N, self.N)
		CM = np.kron(CP, CP) 
		print np.shape(CM.dot(np.ravel(coefficents)))
		# return np.reshape(CM.dot(np.ravel(coefficents)), (self.N+1, self.N+1))

	def projectpatch(self, NB): 
		self.patchval =  self.computepatchvalues(np.pad(self.extractpatchcoeffs(self), \
			(0, NB - self.N), 'constant'))
		return self
	
	def restrictpatch(self, NB): 
		np.set_printoptions(precision=2)
		CM = self.extractpatchcoeffs(self)
		mask = np.zeros(np.shape(CM))
		mask[0:NB, 0:NB] = 1
		RCM = np.multiply(mask, CM)
		self.patchval  =  self.computepatchvalues(RCM)
		return self

	@staticmethod
	def plotpatch(ax, solution, CX, CY, XP, YP, RANGE):

		import matplotlib.pyplot as plt
		from matplotlib import cm
		import matplotlib.patches as patches
		import matplotlib

		# xx, yy = np.meshgrid(XP, YP)
		# ax.plot(xx, yy, 'k.', markersize=0.5)

		minima = np.amin(RANGE)
		maxima = np.amax(RANGE)
		norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
		mapper = cm.ScalarMappable(norm=norm, cmap=cm.viridis)	


		for i in range(len(CX)-1):
			for j in range(len(CY)-1):
				ax.add_patch(patches.Rectangle((CX[i], CY[j]), CX[i+1] - CX[i], CY[j+1] - CY[j], \
					fill=True, facecolor=mapper.to_rgba(solution[j,i])))	# HACK with the indices
				if (0):
					ax.add_patch(patches.Rectangle((CX[i], CY[j]), CX[i+1] - CX[i], CY[j+1] - CY[j], \
					fill=False, linewidth=0.2))
		return ax