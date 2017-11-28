#===============================================================
# Unit tests written to test the Python code
# Soham M 10/2017
#===============================================================

# FIXME
#	> Convergence of two sines or exponentials are always below 10^-14. Is that a problem?
#	> For multiple patches, the boundary values are being set incorrectly.
# 	> For a single patch, we do not find the computed solution to match the analytic solution.
# 	> Decide what the code should return: the patch library or the values.

# TODO
# 	> Missing implementation of futures
# 	> Yet to implement conformal compactification

from scalarwave_util import patch, spec, multipatch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if(1):
	"""
	Test multipatch solution without futures
	"""
	computationaldomain = multipatch(npatches=1, nmodes=20, \
							leftboundary  = lambda x: np.sin(np.pi*x), \
							rightboundary = lambda y: np.sin(np.pi*y), \
							potential 	  = None)
	domain 	 = computationaldomain.globalsolve()
	solution = computationaldomain.assemblegrid(domain) 
	plt.imshow(solution)
	plt.show()

	

