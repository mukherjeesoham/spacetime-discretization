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
computationaldomain = multipatch(npatches=2, nmodes=2, \
						leftboundary  = lambda x: np.exp(-x**2.0/0.1), \
						rightboundary = lambda y: np.exp(-y**2.0/0.1), \
						potential 	  = None)

# call the solver
domain = computationaldomain.globalsolve()

