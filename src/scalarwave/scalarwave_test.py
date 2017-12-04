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

if(0):	# test 1D projection
	x = np.linspace(-1, 1, 1000)
	plt.plot(x, x**3.0, "k--")
	plt.plot(spec.chebnodes(80), projectboundary1D(lambda x: x**3.0, 80), linewidth=0.8)
	plt.show()