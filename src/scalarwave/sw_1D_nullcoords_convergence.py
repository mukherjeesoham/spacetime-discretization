#=============================================================
# Code to investigate convergence and compute energy for a scalar
# wave
#=============================================================

import numpy as np
import futures_utilities as util

def pconv(patchL, patchH):
	Nl  = np.shape(patchL)[0]
	Nh  = np.shape(patchH)[0]
	
	xl  =  util.cheb(Nl-1)[1]
	yl  =  util.cheb(Nl-1)[1]
	Cl  = np.eye(Nl)
	Wl  = np.kron(np.polynomial.chebyshev.chebval(xl, Cl), \
					np.polynomial.chebyshev.chebval(yl, Cl))	
	cfl = np.linalg.solve(Wl, np.ravel(patchL))

	# pad coefficents with zeros
	cfl = np.pad(cfl, (0, int(Nh**2.0 - Nl**2.0)), 'constant')

	xh  = util.cheb(Nh-1)[1]
	yh  = util.cheb(Nh-1)[1]
	Ch  = np.eye(Nh)
	Wh  = np.kron(np.polynomial.chebyshev.chebval(xh, Ch), \
					np.polynomial.chebyshev.chebval(yh, Ch))	
	
	# construct the upscaled patch
	upvec = Wh.dot(cfl)
	upmat = np.resize(upvec, (Nh, Nh))

	# construct the weight matrix
	V = np.outer(util.clencurt(Nh-1), util.clencurt(Nh-1))
	W = np.diag(np.ravel(V))

	# compute the volume integral of the error
	error = patchH - upmat
	L2    =  np.sqrt(np.trace(np.abs(W*np.ravel(error**2.0))))
	return L2

def hconv(grid1, grid2):
	pass

def energy(grid1):
	pass