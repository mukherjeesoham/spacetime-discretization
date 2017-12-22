import unittest
import numpy as np
import sys
sys.path.append('../')
from scalarwave_spectral import spec
from scalarwave_patch import patch

# test projection of 2D functions on grid
np.set_printoptions(precision=4)
x, y   = spec.chebnodes(28), spec.chebnodes(28)
xx, yy = np.meshgrid(x, y) 
zz     = np.exp(-(xx)**2.0) + np.exp(-(yy)**2.0)
f      = lambda x, y: np.exp(-(x)**2.0) + np.exp(-(y)**2.0)
print "- Computing analytic solution."

# project the analytic function
# extremeley restricted in the domain in which this works. FIX IT.
zzP    = spec.projectfunction2D(f, 28, spec.chebnodes(28))	

print "- Finished projecting analytic solution."

# Compute L2 norm error
AS = np.ravel(zz) 	# renaming vars
PS = np.ravel(zzP)
GW = np.ravel(np.outer(spec.chebweights(28), spec.chebweights(28)))
print "+ L2 error nomr:", np.sqrt(np.abs(np.dot(GW, (PS-AS)**2.0)/np.abs(np.dot(GW, AS**2.0))))

if(0):
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure()
	ax  = fig.add_subplot(111, projection='3d')
	ax.plot_wireframe(xx, yy, zz, linewidth=0.8)
	ax.plot_wireframe(xx, yy, zzP, color='k', linewidth=0.8)
	plt.show()