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
if(0):
	computationaldomain = multipatch(npatches=1, nmodes=20, \
						leftboundary   = lambda x: np.exp(-x**2.0/0.1), \
						rightboundary  = lambda y: np.exp(-y**2.0/0.1), \
						potential 	   = None,
						nboundarymodes = 10,
						savefigure     = 1)
else:
    computationaldomain = multipatch(npatches=2, nmodes=40, \
    					leftboundary  = lambda x: np.sin(np.pi*x), \
 	    				rightboundary = lambda y: np.sin(np.pi*y), \
 		    			potential 	  = None, #lambda x, y: np.sin(x+y)*np.exp((-x**2.0)/0.1)*np.exp((-y**2.0)/(0.1)),
 		    			savefigure    = 1)

# call the solver
domain = computationaldomain.globalsolve()
