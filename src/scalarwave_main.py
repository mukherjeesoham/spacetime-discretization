#--------------------------------------------------------------------
# Spacetime Discretization methods Scalar Wave Prototype
# Utilities hanlding patch handling and computation
# Soham M 10-2017
#--------------------------------------------------------------------
from scalarwave_multipatch import multipatch
import numpy as np

"""
# FIXME
	> H-P convergence

# TODO
	> Re-Implement futures
	> Implement conformal compactification
	> Implement polar coordinates
"""

# define a computational domain
computationaldomain = multipatch(npatches=10, nmodes=20, \
						leftboundary  = lambda x: np.exp(-x**2.0/0.1), \
						rightboundary = lambda y: np.exp(-y**2.0/0.1), \
						potential 	  = None)
if(0):
    computationaldomain = multipatch(npatches=8, nmodes=10, \
    					leftboundary  = lambda x: np.sin(0*x), \
 	    				rightboundary = lambda y: np.sin(0*y), \
 		    			potential 	  = lambda x, y: np.sin(x+y)*np.exp((-x**2.0)/0.1)*np.exp((-y**2.0)/(0.1)) )

# call the solver
domain = computationaldomain.globalsolve()
