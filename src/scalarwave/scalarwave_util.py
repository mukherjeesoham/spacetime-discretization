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
	# Function definitions for projecting 1D function on arbitrary points
	#--------------------------------------------------------------------

	@staticmethod
	def vandermonde1D(M, X):
		T    = np.eye(100)
		MX   = np.arange(0, M+1, 1)
		VNDM = np.zeros((len(X), M+1))
		for i, _x in enumerate(X):
			for j, _m in enumerate(MX):
				VNDM[i, j] = np.polynomial.chebyshev.chebval(_x, T[_m])
		return VNDM

	@staticmethod
	def computevalues1D(COEFF, X):
		VNDM = spec.vandermonde1D(len(COEFF)-1, X)
		FN 	 = np.zeros(len(X))
		for i, _x in enumerate(VNDM):
			FN[i] = np.dot(_x, COEFF)
		return FN

	@staticmethod
	def projectfunction1D(function, nmodes, X):
		"""
		Returns M+1 length vector, since one has
		to include T[0, x]
		"""
		M = nmodes
		IP = np.zeros(M+1)
		for m in range(M+1):
			IP[m] = integrate.quadrature(lambda x: function(np.cos(x))*np.cos(m*x), \
				0, np.pi, tol=1.49e-15, rtol=1.49e-15, maxiter=500)[0]

		MX      = np.diag(np.repeat(np.pi/2.0, M+1))
		MX[0]   = MX[0]*2.0
		COEFFS  = np.linalg.solve(MX, IP)
		VALS    = spec.computevalues1D(COEFFS, X)
		return VALS

	#--------------------------------------------------------------------
	# Function definitions for projecting 2D function on arbitrary points
	#--------------------------------------------------------------------

	@staticmethod
	def vandermonde2D(M, X):
		T    = np.eye(100)
		MX   = np.arange(0, M+1, 1)
		VNDM = np.zeros((len(X), M+1))
		for i, _x in enumerate(X):
			for j, _m in enumerate(MX):
				VNDM[i, j] = np.polynomial.chebyshev.chebval(_x, T[_m])
		VNDM2D = np.kron(VNDM, VNDM)
		return VNDM2D

	@staticmethod
	def computevalues2D(COEFF, X):
		VNDM = spec.vandermonde2D(len(COEFF)-1, X)
		FN 	 = np.zeros(len(X))
		for i, _x in enumerate(VNDM):
			FN[i] = np.dot(_x, COEFF)
		return FN

	@staticmethod
	def projectfunction2D(function, nmodes, X):
		# first construct MX; i.e. the RHS of the expression
		M     = nmodes
		MX    = np.diag(np.repeat(np.pi/2.0, M+1))
		MX[0] = MX[0]*2.0
		M2DX  = np.kron(MX, MX)
		IP    = np.zeros((M+1, M+1))
		
		for m in range(M+1):
			for n in range(M+1):
				IP[m, n] = integrate.nquad(lambda x, y: function(np.cos(x), np.cos(y))*np.cos(m*x)*np.cos(n*y), \
					[[-1, 1],[-1, 1]])[0]

		# XXX: Note this is where things may go wrong in the future, due to ravelling.
		COEFFS  = np.linalg.solve(M2DX, np.ravel(IP)) 
		print np.shape(COEFFS)
		VALS    = spec.computevalues2D(COEFFS, X)
		return VALS

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
	def extractpatchBC(PATCHVAL, column):		
		"""
		Returns Boundary of a patch
		"""
		if column == 1:		
			# print "Extracting final column from patch \n", PATCH.patchval				
			return PATCHVAL[:,  -1]
		else:
			# print "Extracting final row from patch \n", PATCH.patchval	
			return PATCHVAL[-1,  :]	

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
	def plotpatchcopy(solution):
		N = int(np.sqrt(np.size(solution)) - 1)
		w = spec.chebweights(N)
 		s = np.insert(np.cumsum(w) - 1, 0, -1)
		xx, yy = np.meshgrid(spec.chebnodes(N), spec.chebnodes(N))

		# FIXME: We are having normalization issues
		# How to normalize the values of the patch to the colorbar?
		znorm = solution/np.amax(solution)

		import matplotlib.pyplot as plt
		from matplotlib import cm
		import matplotlib.patches as patches
		
		colors = cm.viridis(znorm)
		fig    = plt.figure()
		ax     = fig.add_subplot(111, aspect='equal')
		
		ax.set_xlim([-1, 1])
		ax.set_ylim([-1, 1])
		
		ax.plot(xx, yy, 'k.', markersize=0.5)
		
		for i in range(len(s)-1):
			for j in range(len(s)-1):
				ax.add_patch(patches.Rectangle((s[i], s[j]), s[i+1] - s[i], s[j+1] - s[j], \
					fill=True, facecolor=colors[i,j]))
				ax.add_patch(patches.Rectangle((s[i], s[j]), s[i+1] - s[i], s[j+1] - s[j], \
					fill=False, linewidth=0.2))
		plt.show()
		return None

	@staticmethod
	def plotpatch(solution, CX, CY, XP, YP):
		xx, yy = np.meshgrid(XP, YP)

		znorm = solution/np.amax(solution)

		import matplotlib.pyplot as plt
		from matplotlib import cm
		import matplotlib.patches as patches
		
		colors = cm.viridis(znorm)
		fig    = plt.figure()
		ax     = fig.add_subplot(111, aspect='equal')
		
		ax.set_xlim([-1, 1])
		ax.set_ylim([-1, 1])	
		ax.plot(xx, yy, 'k.', markersize=3.5)
		
		for i in range(len(CX)-1):
			for j in range(len(CY)-1):
				ax.add_patch(patches.Rectangle((CX[i], CY[j]), CX[i+1] - CX[i], CY[j+1] - CY[j], \
					fill=True, facecolor=colors[j,i]))
				ax.add_patch(patches.Rectangle((CX[i], CY[j]), CX[i+1] - CX[i], CY[j+1] - CY[j], \
					fill=False, linewidth=0.2))
		plt.show()
		return None

#========================================================================
# Class containing methods to handle and sync multiple patches
#========================================================================
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

	def assemblegrid(self, domain):
		M = self.M
		N = self.N
		I = []
		domain[M-1, M-1]
		for i in range(M):
			J = []
			for j in range(M):  
		  		J.append(domain[i,j])
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
		N = self.N
		S = self.shifts(M)
		X = spec.chebnodes(PATCH.N)
		Y = spec.chebnodes(PATCH.N)
		
		WS = spec.chebweights(N)
 		CC = np.insert(np.cumsum(WS) - 1, 0, -1)

		self.PX = np.sort((X + S[index[0], index[1]][0])/M)
		self.PY = np.sort(-(Y + S[index[0], index[1]][1])/M)

		CX = np.sort((CC + S[index[0], index[1]][0])/M)
		CY = np.sort(-(CC + S[index[0], index[1]][1])/M)

		return CX, CY, self.PX, self.PY

	@staticmethod
	def patchjacobian(M):
		pass

	@staticmethod
	def projectelement():
		pass

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
				# print "XP: ", XP
				# print "YP: ", YP				
				if np.sum(index) == 0:		
					bcol  = spec.projectfunction1D(self.funCOL, 50, YP)
					brow  = spec.projectfunction1D(self.funROW, 50, XP)
				elif (np.prod(index) == 0 and np.sum(index) != 0):	
					if index[1] > index[0]:							
						brow  = spec.projectfunction1D(self.funROW, 50, XP)
						bcol  = patch.extractpatchBC(domain[index[0], index[1]-1], 1)
					else:
						brow = patch.extractpatchBC(domain[index[0]-1,  index[1]], 0)
						bcol = spec.projectfunction1D(self.funCOL, 50, YP)
				else:												
					bcol  = patch.extractpatchBC(domain[index[0], index[1]-1], 1)
					brow  = patch.extractpatchBC(domain[index[0]-1,  index[1]], 0)	

				BC = patch.setBCs(PATCH, BROW = brow, BCOL = bcol, PV = self.computelocalV(XP, YP))
				domain[index[0], index[1]] = PATCH.solve(BC, OPERATOR).patchval	# returns a patch object
				patch.plotpatch(domain[index[0], index[1]], CX, CY, XP, YP)
		return domain


#========================================================================
# Class containing methods to test convergence
#========================================================================
class conv(object):
	"""
	The class contains all the tools to test convergence 
	as we increase the number of patches or the number of 
	modes in each patch. Currently it can only compute 
	convergence.
	"""

	def __init__(self, nmodevec, rightboundary, leftboundary, \
		potential = None, analyticsol = None):
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


