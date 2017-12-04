#===============================================================
# Unit tests written to test the Python code
# Soham M 10/2017
#===============================================================

# FIXME
#	> For multiple patches, the boundary values are being set incorrectly.
# 	> For a single patch, we do not find the computed solution to match the analytic solution.

# TODO
# 	> Missing implementation of futures
# 	> Yet to implement conformal compactification

from scalarwave_util import patch, spec, multipatch, conv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate
import scipy.integrate as integrate

if(0):
	"""
	Test multipatch solution without futures
	"""
	computationaldomain = multipatch(npatches=1, nmodes=20, \
							leftboundary  = lambda x: np.exp(-x**2.0/0.1), \
							rightboundary = lambda y: 0*y, \
							potential 	  = None)
	domain 	 = computationaldomain.globalsolve()
	solution = computationaldomain.assemblegrid(domain) 
	plt.imshow(solution)
	plt.show()

if(0):
	convergencetest = conv(nmodevec = np.arange(3, 10, 2), \
						   leftboundary  = lambda x: np.exp(-x**2.0/0.1), \
						   rightboundary = lambda y: np.exp(-y**2.0/0.1), \
						   potential 	 = None, \
						   analyticsol   = lambda x, y: np.exp(-y**2.0/0.1) + np.exp(-y**2.0/0.1))
	convergencetest.pconv(show=1)

if(0):	# test 1D functions
	npoints = 20
	nmodes  = 20
	f = lambda x: np.exp(-x**2.0/0.1)
	
	x  = spec.chebnodes(npoints)
	v  = spec.projectfunction1D(f, nmodes, npoints)
	s  = f(x)
	print "L2 Error norm: ", np.sqrt(np.abs(np.dot(spec.chebweights(npoints), \
						(v-s)**2.0)/np.abs(np.dot(spec.chebweights(npoints), s**2.0))))
	plt.plot(x, v, 'k.')
	plt.plot(x, f(x), 'm--', linewidth=0.5)
	plt.show()

if(0):	# test 2D functions
	npoints = 9
	nmodes  = 9
	f = lambda x, y: np.exp(-x**2.0/0.1) + np.exp(-y**2.0/0.1)
	x = spec.chebnodes(npoints)
	xx, yy = np.meshgrid(x, x)
	s  = f(xx, yy)
	v  = spec.projectfunction2D(f, nmodes, npoints)
	
