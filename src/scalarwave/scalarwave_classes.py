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

	def __init__(self, N=10, M = 1, loc = np.array([0,0])):
		self.N        = N
		self.M 		  = M
		self.loc 	  = loc
		self.patchval = None
		self.bcmatrix = None

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
		return OP

	def patchgrid(self):
		uu, vv = np.meshgrid(spec.chebnodes(self.N), spec.chebnodes(self.N))
		return uu, vv

	def patchpotential(self, profile):
		PV = np.zeros((self.N+1, self.N+1))
		v = u = spec.chebnodes(self.N)
		shift = np.arange(-self.M + 1, self.M, 2)
		scale = self.M
		for index, val in np.ndenumerate(PV):
			t = (u[index[0]] + v[index[1]])/2.0
			r = (v[index[0]] - u[index[1]])/2.0
			PV[index] = profile(t, r)
		return PV

	def extractpatchBC(self, column):		
		"""
		Returns Boundary of a patch
		"""
		if column == 1:						
			return self.patchval[:,  -1]
		else:
			return self.patchval[-1,  :]	

	def setBCs(self, BROW = None, BCOL = None, fn = None):	
		"""
		Computes the boundary condition + potential array. 
		Note that none of the inputs are necessary to call solve.
		"""
		if fn == None:
			PBC = np.zeros((self.N + 1, self.N + 1))
		else:
			PBC = self.patchpotential(fn)
		PBC = np.reshape(np.multiply(np.diag(self.integrationweights(self)), \
			np.ravel(PBC)), (self.N+1, self.N+1))

		if not (isinstance(BROW, np.ndarray) or isinstance(BCOL, np.ndarray)):
			PBC[0, :] = np.zeros(self.N+1)
			PBC[:, 0] = np.zeros(self.N+1)
		else:
			PBC[0, :] = BROW
			PBC[:, 0] = BCOL
		self.bcmatrix = PBC
		return self.bcmatrix

	def solve(self, boundaryconditions, operator):	
		self.patchval = np.reshape(np.linalg.solve(operator, \
			np.ravel(boundaryconditions)), (self.N+1, self.N+1))
		return self.patchval

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

	#--------------------------------------------------------------------------------
	# Functions to go from global to patch-local coordinates
	#--------------------------------------------------------------------------------
	
	@staticmethod
	def shifts(M):
		L = np.zeros((M, M), dtype=object)
		for k, i in enumerate(range(-M+1, M+1, 2)):
			for l, j in enumerate(range(-M+1, M+1, 2)):
				L[k, l] =  np.array([i, -j])
		return L.T

	def patchtogrid(self, index):
		M = self.M
		S = self.shifts(M)
		X = spec.chebnodes(self.N)
		self.GX = np.sort((X + S[index[0], index[1]][0])/M)
		self.GY = np.sort((X + S[index[0], index[1]][1])/M)
		return self.GX, self.GY

	def gridtopatch(self, index):
		M = self.M
		S = self.shifts(M)
		X = self.GX
		Y = self.GY
		self.PX = (X - S[index[0], index[1]][0])*M
		self.PY = (Y - S[index[0], index[1]][1])*M
		return self.PX, self.PY

	def patchjacobian(M):
		pass

if(0):	# test transformation code
	PATCH  = patch(29, 2)

	X, Y   = patch.patchtogrid(PATCH, [0,0])
	XX, YY = np.meshgrid(X, Y)
	Z00 = np.exp(-np.pi*XX**2.0)*np.sin(np.pi*YY)

	X, Y   = patch.patchtogrid(PATCH, [0,1])
	XX, YY = np.meshgrid(X, Y)
	Z01 = np.exp(-np.pi*XX**2.0)*np.sin(np.pi*YY)

	X, Y   = patch.patchtogrid(PATCH, [1,0])
	XX, YY = np.meshgrid(X, Y)
	Z10 = np.exp(-np.pi*XX**2.0)*np.sin(np.pi*YY)

	X, Y   = patch.patchtogrid(PATCH, [1,1])
	XX, YY = np.meshgrid(X, Y)
	Z11 = np.exp(-np.pi*XX**2.0)*np.sin(np.pi*YY)

	RP = np.block([[Z00, Z01], [Z10, Z11]]) 

	MM, NN = np.meshgrid(spec.chebnodes(59), spec.chebnodes(59))
	FP = np.exp(-np.pi*MM**2.0)*np.sin(np.pi*NN)


	import matplotlib.pyplot as plt
	plt.subplot(1,2,1)
	plt.imshow(RP)
	plt.subplot(1,2,2)
	plt.imshow(FP)
	plt.show()

class conv(object):
	"""
	The class contains all the tools to test convergence 
	as we increase the number of patches or the number of 
	modes in each patch. Currently it can only compute 
	convergence.
	"""
	@staticmethod
	def pconv(NP, BROW = None, BCOL = None, FN=None, SOL=None):
		"""
		For convergence tests we can only pass functional forms of BROW, BCOL 
		and the potential
		"""
		print 60*'='
		print "==> Testing p-convergence"
		print 60*'='
		if SOL == None:
			print "   - No functional form of the solution is provided."
			print "   - Computing relative L2 norm of the error."
		else:
			print "   - Functional form of the solution is provided."
			print "   - Computing L2 norm of the error."

		PATCHES = []
		ERROR   = []

		for index, p in enumerate(NP):
			PATCH = patch(p)
			print "   + Computing patch for N = %r"%(p)
			patch.solve(PATCH, \
			  			patch.setBCs(PATCH, BROW(patch.chebnodes(PATCH)), BCOL(patch.chebnodes(PATCH)), FN), 
			  			patch.operator(PATCH))
			PATCHES.append(PATCH)
			if SOL == None: 
				if index !=0:
					W 	= np.diag(patch.integrationweights(PATCHES[index]))
					S 	= np.ravel(PATCHES[index].patchval)
					C 	= np.ravel(patch.projectpatch(PATCHES[index-1], NP[index]))
					RL2 = np.sqrt(np.abs(np.dot(W, (S-C)**2.0)/np.abs(np.dot(W, S))))
					print "   \t \t RL2: ", RL2
					ERROR.append(RL2)
			else:
				W 	= np.diag(patch.integrationweights(PATCHES[index]))
				S 	= np.ravel(PATCHES[index].patchval)
				XX, YY = np.meshgrid(patch.chebnodes(PATCH), patch.chebnodes(PATCH))
				C = np.ravel(SOL(XX, YY))

				L2 = np.sqrt(np.abs(np.dot(W, (S-C)**2.0)))
				print "   \t \t L2: ", L2
				ERROR.append(L2)
	
		print "   - Finished computing L2 norm. Saving results..."	
		import matplotlib.pyplot as plt
		with plt.rc_context({ "font.size": 20., "axes.titlesize": 20., "axes.labelsize": 20., \
         "xtick.labelsize": 20., "ytick.labelsize": 20., "legend.fontsize": 20., \
         "figure.figsize": (20, 12), \
		 "figure.dpi": 300, "savefig.dpi": 300, "text.usetex": True}):
			if SOL == None:
				plt.semilogy(NP[1:], ERROR, 'm-o')
			else:
				plt.semilogy(NP, ERROR, 'm-o')
			plt.xlabel(r"$N(p)$")
			plt.ylabel(r"$L_2~\rm{norm}~(\rm{Log~scale})$")
			plt.grid()
			plt.title(r"$\rm{Convergence~for~(p)~refinment}$")
			plt.savefig("./output/p-conv.pdf", bbox_inches='tight')
			plt.close()
		print "Done."
		return None
