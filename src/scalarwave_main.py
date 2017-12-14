#--------------------------------------------------------------------
# Spacetime Discretization methods Scalar Wave Prototype 
# Utilities hanlding patch handling and computation
# Soham M 10-2017
#--------------------------------------------------------------------
from scalarwave_multipatch import multipatch
import numpy as np

"""
# FIXME
	> For multiple patches, the boundary values are being set incorrectly.
	> For a single patch, we do not find the computed solution to match the
	> analytic solution for exponential incoming waves

# TODO
	> Missing implementation of futures
	> Yet to implement conformal compactification
	> Implement polar coordinates
"""

# define a computational domain
computationaldomain = multipatch(npatches=1, nmodes=20, \
						leftboundary  = lambda x: np.exp(-x**2.0/0.1), \
						rightboundary = lambda y: np.exp(-y**2.0/0.1), \
						potential 	  = None)

computationaldomain = multipatch(npatches=8, nmodes=10, \
						leftboundary  = lambda x: np.sin(0*x), \
						rightboundary = lambda y: np.sin(0*y), \
						potential 	  = lambda x, y: np.sin(x+y)*np.exp((-x**2.0)/0.1)*np.exp((-y**2.0)/(0.1)) )

# call the solver
domain = computationaldomain.globalsolve()

