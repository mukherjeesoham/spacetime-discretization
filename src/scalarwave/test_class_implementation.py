#===============================================================
# Tests for scalar wave equation in Minkowski Spacetime
# Soham M 10/2017
#===============================================================

import numpy as np
from scalarwave_classes import patch, spec, conv

if(1):
	# create and solve for different boundary conditions 
	# on a single patch
	PATCH = patch(40)
	X 	  = patch.chebnodes(PATCH)
	print "Finished computing the nodes on the patch."
	OP	  = patch.operator(PATCH)
	print "Finished computing the operator"
	# BC 	  = patch.setBCs(PATCH, BROW = np.exp(-np.pi*X**2.0), BCOL = np.exp(-np.pi*X**2.0))
	BC 	  = patch.setBCs(PATCH, BROW = np.sin(np.pi*X), BCOL = np.sin(np.pi*X))
	BC 	  = patch.setBCs(PATCH, fn = lambda X, Y: np.sin(X)*np.exp((-X**2.0)/0.1)*np.exp((-Y**2.0)/(0.1)))
	print "Finished computing BCs"
	VAL   = patch.solve(PATCH, BC, OP)
	print "Finished solving the linear equation"
	patch.plotpatch(PATCH)
else:
	# compute convergence for resolutions in NP
	NP = np.arange(12, 40, 1)
	# conv.pconv(NP, BROW = lambda X: np.exp(-np.pi*X**2.0), BCOL = lambda X: np.exp(-np.pi*X**2.0))
	# conv.pconv(NP, BROW = lambda X: np.sin(np.pi*X), BCOL = lambda X: np.sin(np.pi*X))
	conv.pconv(NP, FN = lambda X, Y: np.sin(X)*np.exp((-X**2.0)/0.1)*np.exp((-Y**2.0)/(0.1)))
