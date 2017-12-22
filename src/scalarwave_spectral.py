#--------------------------------------------------------------------
# Spacetime Discretization methods Scalar Wave Prototype 
# Utilities for spectral integration and differentiation
# Soham M 10-2017
#--------------------------------------------------------------------

import numpy as np
from scipy import integrate

class spec(object):
	@staticmethod
	def chebnodes(N):
		"""
		see <https://github.com/UBC-Astrophysics/Spectral>
		"""
		if (N != 0):
			return np.cos(np.pi*np.arange(0,N+1)/N)
		else:
			return np.array([0])

	@staticmethod
	def chebmatrix(N):
		"""
		see <https://github.com/UBC-Astrophysics/Spectral>
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
	def vandermonde(N, X):
		T    = np.eye(100)
		MX   = np.arange(0, N+1, 1)
		VNDM = np.zeros((len(X), N+1))
		
		for i, _x in enumerate(X):
			for j, _m in enumerate(MX):
				VNDM[i, j] = np.polynomial.chebyshev.chebval(_x, T[_m])
		return VNDM

	@staticmethod
	def computevalues1D(COEFF, X):
		VNDM = spec.vandermonde(len(COEFF)-1, X)
		FN 	 = np.zeros(len(X))
		for i, _x in enumerate(VNDM):
			FN[i] = np.dot(_x, COEFF)
		return FN

	@staticmethod
	def projectfunction1D(function, nmodes, X):
		"""
		Projects analytic function on boundaries.
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

	@staticmethod
	def computevalues2D(COEFF, X):
		VNDM   = spec.vandermonde(len(X)-1, X)
		VNDM2D = np.kron(VNDM, VNDM)
		FN 	   = np.zeros(len(X)**2)
		for i, _x in enumerate(VNDM2D):
			FN[i] = np.dot(_x, np.ravel(COEFF))
		return np.reshape(FN, (len(X), len(X)))

	@staticmethod
	def projectfunction2D(function, nmodes, X):
		"""
		Project a 2D function (potential) on a patch
		"""
		M     = nmodes
		MX    = np.diag(np.repeat(np.pi/2.0, M+1))
		MX[0] = MX[0]*2.0
		M2DX  = np.kron(MX, MX)
		IP    = np.zeros((M+1, M+1))
		for m in range(M+1):
			for n in range(M+1):
				I = integrate.nquad(lambda x, y: function(np.cos(x), np.cos(y))*np.cos(m*x)*np.cos(n*y), \
					[[0, np.pi],[0, np.pi]], opts={"epsabs": 1e-15})
				IP[m,n] = I[0]

		# FIXME: Perhaps we are doing this wrong
		COEFFS  = np.linalg.solve(M2DX, np.ravel(IP)) 


		PSOL = np.zeros((computationaldomain.N*2 + 2, computationaldomain.N*2 + 2))
		for i, _x in enumerate(XNEW):		# all points in x-dir
			for j, _y in enumerate(YNEW):	# all points in y-dir			
				SUM = 0
				T   = np.eye(100)
				for index, value in np.ndenumerate(COEFFS):	# sum over all coefficents
					k  = index[0]
					l  = index[1]
					Tk_x = np.polynomial.chebyshev.chebval(_x, T[k])
					Tl_y = np.polynomial.chebyshev.chebval(_y, T[l])
					SUM = SUM + COEFFS[index]*Tk_x*Tl_y
				PSOL[i,j] = SUM
		return PSOL

