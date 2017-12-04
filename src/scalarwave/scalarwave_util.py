#========================================================================
# Utilities for scalar wave equation in Minkowski Spacetime
# Soham M 10/2017
#========================================================================

import numpy as np
from scipy import integrate
from scipy.integrate import quad, dblquad

#========================================================================
# Class containing basic methods
#========================================================================
class spec(object):

	#--------------------------------------------------------------------
	# Function definitions for spectral integration and differentiation
	#--------------------------------------------------------------------
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
			return np.array([0])

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

	#--------------------------------------------------------------------
	# Function definitions for projecting in the 1D case
	#--------------------------------------------------------------------

	@staticmethod
	def vandermonde1D(M, N):
		T    = np.eye(100)
		X    = spec.chebnodes(N)
		MX   = np.arange(0, M+1, 1)
		VNDM = np.zeros((N+1, M+1))
		for i, _x in enumerate(X):
			for j, _m in enumerate(MX):
				VNDM[i, j] = np.polynomial.chebyshev.chebval(_x, T[_m])
		return VNDM

	@staticmethod
	def computevalues1D(COEFF, N):
		VNDM = spec.vandermonde1D(len(COEFF)-1, N)
		FN 	 = np.zeros(N+1)
		for i, _x in enumerate(VNDM):
			FN[i] = np.dot(_x, COEFF)
		return FN

	@staticmethod
	def projectfunction1D(function, nmodes, npoints):
		"""
		Returns M+1 length vector, since one has
		to include T[0, x]
		"""
		M = nmodes
		C = np.zeros(M+1)
		for m in range(M+1):
			C[m] = integrate.quadrature(lambda x: function(np.cos(x))*np.cos(m*x), \
				0, np.pi, tol=1.49e-15, rtol=1.49e-15, maxiter=500)[0]

		MX      = np.diag(np.repeat(np.pi/2.0, M+1))
		MX[0]   = MX[0]*2.0
		COEFFS  = np.linalg.solve(MX, C)
		VALS    = spec.computevalues1D(COEFFS, npoints)
		return VALS

	#--------------------------------------------------------------------
	# Function definitions for projecting in the 2D case
	#--------------------------------------------------------------------

	@staticmethod
	def vandermonde2D(M, N):
		pass

	@staticmethod
	def computevalues2D(COEFF, N):
		pass

	@staticmethod
	def projectfunction2D(function, nmodes, npoints):
		"""
		Returns M+1 length vector, since one has
		to include T[0, x]
		"""
		M = nmodes
		C = np.zeros((M+1, M+1))
		for m in range(M+1):
			for n in range(M+1):
				C[m, n] = integrate.nquad(lambda x, y: function(np.cos(x), np.cos(y))*np.cos(m*x)*np.cos(n*y), \
					[[-1, 1],[-1, 1]])[0]
		
		return None

	@staticmethod
	def createfilter():
		pass

#========================================================================
# Class containing methods to handle a single patch
#========================================================================
class patch(spec):

	def __init__(self, N, loc = None):
		self.N        = N
		self.loc 	  = loc

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
		CP = spec.vandermonde1D(self.N, self.N)
		CM = np.kron(CP, CP)  
		return np.reshape(np.linalg.solve(CM, np.ravel(self.patchval)), (self.N+1, self.N+1))

	def computepatchvalues(self, coefficents):
		CP = spec.vandermonde1D(self.N, self.N)
		CM = np.kron(CP, CP) 
		print np.shape(CM.dot(np.ravel(coefficents)))
		# return np.reshape(CM.dot(np.ravel(coefficents)), (self.N+1, self.N+1))

	def projectpatch(self, NB): 
		self.patchval =  self.computepatchvalues(np.pad(self.extractpatchcoeffs(self), (0, NB - self.N), 'constant'))
		return self
	
	def restrictpatch(self, NB): 
		np.set_printoptions(precision=2)
		CM = self.extractpatchcoeffs(self)
		mask = np.zeros(np.shape(CM))
		mask[0:NB, 0:NB] = 1
		RCM = np.multiply(mask, CM)
		self.patchval  =  self.computepatchvalues(RCM)
		return self

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

	@staticmethod
	def patchjacobian(M):
		pass

	@staticmethod
	def projectelement():
		pass

	def globalsolve(self):
		domain = np.zeros((self.M, self.M), dtype=object)
		grid   = self.makeglobalgrid(self.M)	
		PATCH  = patch(self.N)

		# Compute the Jacobian and correct for the operator
		OPERATOR = patch.operator(PATCH)

		for i in range(int(np.max(grid))+1):
			slice = np.transpose(np.where(grid==i))
			for index in slice:
				PATCH.loc = index
				if not self.M == 1:
					print "Computing patch: ", index
				XP, YP = self.gridtopatch(PATCH, index) 	
						
				if np.sum(index) == 0:	# initial patch		
					bcol  = self.funcLB(YP)
					brow  = self.funcRB(XP)
				elif (np.prod(index) == 0 and np.sum(index) != 0):	
					if index[0] > index[1]:							
						bcol  = self.funcLB(YP)
						brow  = patch.extractpatchBC(domain[index[0]-1, index[1]], 0)	
					else:	
						bcol  = patch.extractpatchBC(domain[index[0], index[1]-1], 1)												
						brow  = self.funcRB(XP)
				else:												
					bcol  = patch.extractpatchBC(domain[index[0], index[1]-1], 1)
					brow  = patch.extractpatchBC(domain[index[0]-1, index[1]], 0)

				BC = patch.setBCs(PATCH, BROW = brow, BCOL = bcol, PV = self.computelocalV(XP, YP))
				domain[index[0], index[1]] = PATCH.solve(BC, OPERATOR)	# returns a patch object

		return domain


class conv(object):
	"""
	The class contains all the tools to test convergence 
	as we increase the number of patches or the number of 
	modes in each patch. Currently it can only compute 
	convergence.
	"""

	def __init__(self, nmodevec, rightboundary, leftboundary, potential = None, analyticsol = None):
		self.nmodevec	   = nmodevec
		self.rightboundary = rightboundary
		self.leftboundary  = leftboundary
		self.funcV	       = potential 
		self.analyticsol   = analyticsol

	@staticmethod
	def projectanalyticsolutiononpatch(exactsolfunc, N):
		x, y  = np.meshgrid(spec.chebnodes(N), spec.chebnodes(N))
		PATCH = patch(N)
		PATCH.patchval = exactsolfunc(x,y)
		return PATCH.projectpatch(N)

	@staticmethod
	def computeLXerror(PCS, PAS, error="L2"):
		CS = np.ravel(PCS.patchval)
		AS = np.ravel(np.fliplr(np.flipud(PAS.patchval)))
		GW = np.diag(patch.integrationweights(PAS))
		
		if error == "L2":
			ERROR = np.sqrt(np.abs(np.dot(GW, (CS-AS)**2.0)/np.abs(np.dot(GW, AS**2.0))))
		elif error == "L1":
			ERROR = np.sqrt(np.abs(np.dot(GW, (CS-AS))/np.abs(np.dot(GW, AS))))
		elif error == "Linf":
			ERROR = np.amax(np.abs(CS-AS))
		else:
			raise ValueError('Do not know which error to compute.')
		return ERROR

	@staticmethod
	def computeL2forPCS(PCS, nmodes, error="L2"):
		CS = np.ravel(PCS.patchval)
		AS = np.ravel(PCS.restrictpatch(nmodes).patchval)
		GW = np.diag(patch.integrationweights(PCS))
		ERROR = np.sqrt(np.abs(np.dot(GW, (CS-AS)**2.0)/np.abs(np.dot(GW, AS**2.0))))
		return ERROR

	def pconv(self, show):
		"""
		For convergence tests we only pass functional forms of BROW, BCOL 
		and the potential
		"""
		print 60*'='
		print "==> Testing p-convergence"
		print 60*'='

		ERROR = np.zeros(np.size(self.nmodevec))

		for index, nmodes in enumerate(self.nmodevec):

			print "Computing for N = %r" %(nmodes)
			computationaldomain = multipatch(npatches=1, nmodes=nmodes, \
							leftboundary  = self.leftboundary,
							rightboundary = self.rightboundary,
							potential 	  = self.funcV)
			domain = computationaldomain.globalsolve()

			PCS = domain[0,0].projectpatch(nmodes*2)
			PAS = self.projectanalyticsolutiononpatch(self.analyticsol, nmodes*2)
			ERROR[index] = self.computeLXerror(PCS, PAS, error = "L2")
			print "   - L2 = %e"%ERROR[index]

		print "   - Finished computing L2 norm. Saving results..."		

		if(show):
			import matplotlib.pyplot as plt
			plt.semilogy(self.nmodevec, ERROR, 'm-o')
			plt.xticks(self.nmodevec)
			plt.xlabel(r"$N(p)$")
			plt.title(r"$L_2~\rm{norm}~(\rm{Log~scale})$")
			plt.grid()
			plt.show()
		print "Done."


