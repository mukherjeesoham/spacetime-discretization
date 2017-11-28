#===============================================================
# Utilities for scalar wave equation in Minkowski Spacetime
# Soham M 10/2017
#===============================================================

import numpy as np

class spec(object):

	def __init__(self, N):
		self.N = N

	@staticmethod
	def chebnodes(N):
		"""
		see <https://github.com/UBC-Astrophysics/Spectral>
		Compute the Chebyshev differentiation matrices 
		and nodes in a direction given the resolution
		local to a patch.
		"""
		if (N != 0):
			return np.cos(np.pi*np.arange(0,N+1)/N)
		else:
			raise ValueError('Number of points cannot be zero!')

	@staticmethod
	def chebmatrix(N):
		"""
		see <https://github.com/UBC-Astrophysics/Spectral>
		Compute the Chebyshev differentiation matrices 
		and nodes in a direction given the resolution
		local to a patch.
		"""
		if (N != 0):
			x  = np.cos(np.pi*np.arange(0,N+1)/N)
			c  = np.concatenate(([2],np.ones(N-1),[2]))*(-1)**np.arange(0,N+1)
			X  = np.tile(x,(N+1,1))
			dX = X-np.transpose(X)
			c  = np.transpose(c)
			D  = -np.reshape(np.kron(c,1/c),(N+1,N+1))/(dX+np.eye(N+1))
			return D - np.diagflat(np.sum(D,axis=1))
		else:
			raise ValueError('Number of points cannot be zero!')

	@staticmethod
	def chebweights(N):
		"""
		see <https://github.com/mikaem/spmpython>
		CLENCURT nodes x (Chebyshev points) and weights w 
		for Clenshaw-Curtis quadrature
		"""
		theta = np.pi*np.arange(0,N+1)/N
		x   = np.cos(theta)
		W   = np.zeros(N+1)
		ind = np.arange(1,N)
		v   = np.ones(N-1)
		if np.mod(N,2)==0:
			W[0] = 1./(N**2-1)
			W[N] = W[0]
			for k in np.arange(1,int(N/2.)):
				v = v-2*np.cos(2*k*theta[ind])/(4*k**2-1)
			v = v - np.cos(N*theta[ind])/(N**2-1)
		else:
			W[0] = 1./N**2
			W[N] = W[0]
			for k in np.arange(1,int((N-1)/2.)+1):
		   		v = v-2*np.cos(2*k*theta[ind])/(4*k**2-1)
		W[ind] = 2.0*v/N
		return W

	@staticmethod
	def createfilter():
		pass

	@staticmethod
	def computechebpoly():
		pass

class patch(spec):

	def __init__(self, N, loc = None):
		self.N        = N
		self.loc 	  = loc

	@staticmethod
	def chebnodes(self):
		"""
		Compute the integration weight matrix for the 2D grid
		"""
		N = self.N
		return spec.chebnodes(N)

	@staticmethod
	def integrationweights(self):
		"""
		Compute the integration weight matrix for the 2D grid
		"""
		N = self.N
		return np.diag(np.ravel(np.outer(spec.chebweights(N), \
			spec.chebweights(N))))

	@staticmethod
	def operator(self):
		# TODO: Optimize this computation. Can we construct OP element-wise?
		# Also, inherit previously computed quantities.
		N = self.N
		DU = np.kron(spec.chebmatrix(N), np.eye(N+1))
		DV = np.kron(np.eye(N+1), spec.chebmatrix(N))
		D  = np.dot(DU,DV) + np.dot(DV,DU)
		OP = self.integrationweights(self).dot(D)	# XXX: We do not symmetrize the operator!
		# Replace rows in the matrix to implement Dirichlet boundary
		# conditions TODO: Implement this in the physics using Lagrange multipliers?
		BC = np.zeros((N+1,N+1))
		BC[0, :] = BC[:, 0] = 1 # Set Dirichlet BCs at adjacent edges
		OP[np.where(np.ravel(BC)==1)[0]] = np.eye((N+1)**2)[np.where(np.ravel(BC)==1)[0]]  
		self.operator = OP
		return OP

	@staticmethod
	def eigenval(self):
		print "==> Computing eigenvalues"
		eigenvalues = np.linalg.eigvals(self.operator)
		emax = np.amax(np.abs(eigenvalues))
		emin = np.amin(np.abs(eigenvalues))
		print "   - Eigenval (max/min): ", emax/emin
		return eigenvalues
  
	def patchgrid(self):
		uu, vv = np.meshgrid(spec.chebnodes(self.N), spec.chebnodes(self.N))
		return uu, vv

	@staticmethod
	def extractpatchBC(PATCH, column):		
		"""
		Returns Boundary of a patch
		"""
		if column == 1:						
			return PATCH.patchval[:,  -1]
		else:
			return PATCH.patchval[-1,  :]	

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
		PBC = np.reshape(np.multiply(np.diag(self.integrationweights(self)), \
							np.ravel(PBC)), (self.N+1, self.N+1))

		PBC[0, :] = BROW
		PBC[:, 0] = BCOL

		self.bcmatrix = PBC
		return self.bcmatrix

	def solve(self, boundaryconditions, operator):	
		self.patchval = np.reshape(np.linalg.solve(operator, \
			np.ravel(boundaryconditions)), (self.N+1, self.N+1))
		return self

	@staticmethod
	def extractpatchcoeffs(self):
		CP = np.polynomial.chebyshev.chebval(spec.chebnodes(self.N), np.eye(self.N+1))
		CM = np.kron(CP, CP)  
		return np.reshape(np.linalg.solve(CM, np.ravel(self.patchval)), (self.N+1, self.N+1))

	@staticmethod
	def extractpatchvalues(coefficents):
		N = int(np.sqrt(np.size(coefficents)))-1
		CP = np.polynomial.chebyshev.chebval(spec.chebnodes(N), np.eye(N+1))
		CM = np.kron(CP, CP) 
		return np.reshape(CM.dot(np.ravel(coefficents)), (N+1, N+1))

	def projectpatch(self, NB = 0): 
		return self.extractpatchvalues(np.pad(self.extractpatchcoeffs(self), (0, NB -
		self.N), 'constant'))
	
	def plotpatch(self):
		import matplotlib.pyplot as plt
		import scipy
		from scipy import ndimage
		with plt.rc_context({ "font.size": 20., "axes.titlesize": 20., "axes.labelsize": 20., \
         "xtick.labelsize": 20., "ytick.labelsize": 20., "legend.fontsize": 20., \
         "figure.figsize": (20, 12), \
		 "figure.dpi": 300, "savefig.dpi": 300, "text.usetex": True}):
			plt.imshow(np.flipud(self.patchval))
			plt.colorbar()
			plt.savefig("./output/patch-solution.pdf", bbox_inches='tight')
			plt.axis('off')
			plt.close()
		return None

class multipatch(object):
	"""
	The main function, which after futurization,
	takes the size and number of patches, the boundary conditions/the potential
	and computes, and then assembles the entire solution
	"""
	def __init__(self, npatches, nmodes, leftboundary, rightboundary, potential):
		self.M      = npatches
		self.N 		= nmodes
		self.funcLB = leftboundary
		self.funcRB = rightboundary
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

	def assemblegrid(self, domain):
		M = self.M
		N = self.N
		I = []
		domain[M-1, M-1]
		for i in range(M):
			J = []
			for j in range(M):  
		  		J.append(domain[i,j].patchval)
			I.append(J)
  
		blocks 	= np.block(I)
		columns = np.linspace(0, M*(N+1), M+1)[1:-1]
		blocks 	= np.delete(blocks, columns, 0) 
		blocks 	= np.delete(blocks, columns, 1) 
		return blocks

	def computelocalV(self, XP, YP):
		XX, YY = np.meshgrid(XP, YP)
		if self.funcV == None:
			return (XX + YY)*0
		else:
			return self.funcV(XX, YY)

	def gridtopatch(self, PATCH, index):
		M = self.M
		S = self.shifts(M)
		X = spec.chebnodes(PATCH.N)
		Y = spec.chebnodes(PATCH.N)
		self.PX = np.sort((X + S[index[0], index[1]][0])/M)
		self.PY = np.sort(-(Y + S[index[0], index[1]][1])/M)
		return self.PX, self.PY

	def patchjacobian(M):
		pass

	def globalsolve(self):
		domain = np.zeros((self.M, self.M), dtype=object)
		grid   = self.makeglobalgrid(self.M)	
		PATCH  = patch(self.N)

		# FIXME: Hacky way to compute the Jacobian
		OPERATOR = patch.operator(PATCH)/self.M

		for i in range(int(np.max(grid))+1):
			slice = np.transpose(np.where(grid==i))
			for index in slice:
				PATCH.loc = index
				print "Computing patch: ", index
				XP, YP = self.gridtopatch(PATCH, index) 							
				if np.sum(index) == 0:	# initial patch		
					bcol  = self.funcLB(YP)
					brow  = self.funcLB(XP)
				elif (np.prod(index) == 0 and np.sum(index) != 0):	
					if index[0] > index[1]:									
						bcol  = self.funcLB(YP)
						brow  = patch.extractpatchBC(domain[index[0]-1,index[1]], 0)	
					else:	
						bcol  = patch.extractpatchBC(domain[index[0],index[1]-1], 1)												
						brow  = self.funcLB(XP)
				else:												
					bcol  = patch.extractpatchBC(domain[index[0],index[1]-1], 1)
					brow  = patch.extractpatchBC(domain[index[0]-1,index[1]], 0)

				BC = patch.setBCs(PATCH, BROW = brow, BCOL = bcol, PV = self.computelocalV(XP, YP))
				domain[index[0],index[1]] = patch.solve(PATCH, BC, OPERATOR)

		self.patchlibrary  = domain
		self.globalsol = self.assemblegrid(self.patchlibrary)
		return self.globalsol



