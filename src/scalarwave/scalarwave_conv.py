
# FIXME: This is screwed up
class conv(object):
	"""
	The class contains all the tools to test convergence 
	as we increase the number of patches or the number of 
	modes in each patch. Currently it can only compute 
	convergence.
	"""

	def __init__(self, NP = None, BROWfn = None, BCOLfn = None, FN = None, SOL = None):
		self.NP = NP
		self.browfunc = BROWfn
		self.bcolfunc = BCOLfn
		self.fn       = FN 
		self.sol      = SOL

	# FIXME: Not recognizing as a method of the class
	@staticmethod
	def plotpconv(NP, ERROR):
		import matplotlib.pyplot as plt
		with plt.rc_context({ "font.size": 20., "axes.titlesize": 20., "axes.labelsize": 20., \
		"xtick.labelsize": 20., "ytick.labelsize": 20., "legend.fontsize": 20., \
		"figure.figsize": (20, 12), \
		"figure.dpi": 300, "savefig.dpi": 300, "text.usetex": True}):
			plt.semilogy(NP, ERROR, 'm-o')
			plt.xticks(NP)
			plt.xlabel(r"$N(p)$")
			plt.ylabel(r"$L_2~\rm{norm}~(\rm{Log~scale})$")
			plt.grid()
			plt.title(r"$\rm{Convergence~for~(p)~refinment}$")
			plt.savefig("./output/p-conv.pdf", bbox_inches='tight')
			plt.close()
		return None

	# FIXME: Not recognizing as a method of the class
	@staticmethod
	def computepatches(NP, BROWfn, BCOLfn, FN):
		PATCHES = np.zeros(len(NP), dtype = object)
		for index, p in enumerate(NP):
			PATCH = patch(p)
			print "   + Computing patch for N = %r"%(p)
			if (BROWfn == None) or (BCOLfn == None):
				patch.solve(PATCH, \
			  				patch.setBCs(PATCH, fn = FN), \
			  				patch.operator(PATCH))
			else:
				patch.solve(PATCH, \
			  				patch.setBCs(PATCH, 
			  					BROW = BROWfn(patch.chebnodes(PATCH)), \
			  					BCOL = BCOLfn(patch.chebnodes(PATCH)), 
			  					fn = FN), \
			  				patch.operator(PATCH))
			PATCHES[index] = PATCH
		return PATCHES

	# FIXME: Not recognizing as a method of the class
	@staticmethod
	def computL2error_sol(NP, BROWfn, BCOLfn, FN, SOL):
		PATCHES = computepatches(NP, BROWfn, BCOLfn, FN)
		ERROR = np.zeros(np.size(NP), dtype = object)
		for index, p in enumerate(NP):
			XX, YY = np.meshgrid(patch.chebnodes(PATCHES[index]), patch.chebnodes(PATCHES[index]))
			S  = np.ravel(patch.projectpatch(SOL(XX, YY), NP[index]))
			C  = np.ravel(PATCHES[index])
			W  = np.diag(patch.integrationweights(PATCHES[index]))
			L2 = np.sqrt(np.abs(np.dot(W, (S-C)**2.0)/np.abs(np.dot(W, S**2.0))))
			print "   + [N = %r] L2: %e" %(NP[index], L2)
			ERROR[index] = L2
		return ERROR


	# FIXME: Not recognizing as a method of the class
	@staticmethod
	def computL2error_nosol(NP, BROWfn, BCOLfn, FN):
		PATCHES = computepatches(NP, BROWfn, BCOLfn, FN)
		ERROR = np.zeros(np.size(NP), dtype = object)
		for index, p in enumerate(NP):
			S  = np.ravel(PATCHES[-1].patchval)
			C  = np.ravel(patch.projectpatch(PATCHES[index], NP[-1]))
			W  = np.diag(patch.integrationweights(PATCHES[-1]))
			L2 = np.sqrt(np.abs(np.dot(W, (S-C)**2.0)/np.abs(np.dot(W, S**2.0))))
			print "   + [N = %r] L2: %e" %(NP[index], L2)
			ERROR[index] = L2
		return ERROR

	def pconv(self, show):
		"""
		For convergence tests we can only pass functional forms of BROW, BCOL 
		and the potential
		"""

		NP = self.NP
		BROWfn = self.browfunc
		BCOLfn = self.bcolfunc
		FN = self.fn
		SOL= self.sol
		print 60*'='
		print "==> Testing p-convergence"
		print 60*'='

		PATCHES = np.zeros(len(NP), dtype = object)
		for index, p in enumerate(NP):
			PATCH = patch(p)
			print "   + Computing patch for N = %r"%(p)
			if (BROWfn == None) or (BCOLfn == None):
				patch.solve(PATCH, \
			  				patch.setBCs(PATCH, fn = FN), \
			  				patch.operator(PATCH))
			else:
				patch.solve(PATCH, \
			  				patch.setBCs(PATCH, 
			  					BROW = BROWfn(patch.chebnodes(PATCH)), \
			  					BCOL = BCOLfn(patch.chebnodes(PATCH)), 
			  					fn = FN), \
			  				patch.operator(PATCH))
			PATCHES[index] = PATCH
		
		ERROR = np.zeros(np.size(NP), dtype = object)
		if SOL == None:
			print "   - No functional form of the solution is provided."
			print "   - Computing relative L2 norm of the error."
				
			for index, p in enumerate(NP):
				S  = np.ravel(PATCHES[-1].patchval)
				C  = np.ravel(patch.projectpatch(PATCHES[index], NP[-1]))
				W  = np.diag(patch.integrationweights(PATCHES[-1]))
				L2 = np.sqrt(np.abs(np.dot(W, (S-C)**2.0)/np.abs(np.dot(W, S**2.0))))
				if not index == len(NP) - 1:		
					print "   + [N = %r] L2: %e" %(NP[index], L2)
				ERROR[index] = L2
			NP    = NP[:-1] 
			ERROR = ERROR[:-1] 
		else:
			print "   - Functional form of the solution is provided."
			print "   - Computing L2 norm of the error."
			ERROR = np.zeros(np.size(NP), dtype = object)
			for index, p in enumerate(NP):
				XX, YY = np.meshgrid(patch.chebnodes(PATCHES[index]), patch.chebnodes(PATCHES[index]))
				PATCH = patch(p)
				PATCH.patchval = SOL(XX, YY)
				S  = np.ravel(patch.projectpatch(PATCH, NP[index]))
				C  = np.ravel(PATCHES[index])
				W  = np.diag(patch.integrationweights(PATCHES[index]))
				L2 = np.sqrt(np.abs(np.dot(W, (S-C)**2.0)/np.abs(np.dot(W, S**2.0))))
				print "   + [N = %r] L2: %e" %(NP[index], L2)
				ERROR[index] = L2
		return ERROR

		print "   - Finished computing L2 norm. Saving results..."		

		import matplotlib.pyplot as plt
		if show==0:
			with plt.rc_context({ "font.size": 20., "axes.titlesize": 20., "axes.labelsize": 20., \
			"xtick.labelsize": 20., "ytick.labelsize": 20., "legend.fontsize": 20., \
			"figure.figsize": (20, 12), \
			"figure.dpi": 300, "savefig.dpi": 300, "text.usetex": True}):
				plt.semilogy(NP, ERROR, 'm-o')
				plt.xticks(NP)
				plt.xlabel(r"$N(p)$")
				plt.ylabel(r"$L_2~\rm{norm}~(\rm{Log~scale})$")
				plt.grid()
				plt.title(r"$\rm{Convergence~for~(p)~refinment}$")
				plt.savefig("./output/p-conv.pdf", bbox_inches='tight')
				plt.close()
		else:
			plt.semilogy(NP, ERROR, 'm-o')
			plt.xticks(NP)
			plt.xlabel(r"$N(p)$")
			plt.ylabel(r"$L_2~\rm{norm}~(\rm{Log~scale})$")
			plt.grid()
			plt.title(r"$\rm{Convergence~for~(p)~refinment}$")
			plt.show()
		print "Done."
