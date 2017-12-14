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
		VALS    = spec.computevalues2D(COEFFS, X)
		return VALS

