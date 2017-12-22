import unittest
import numpy as np
import sys
sys.path.append('../')
from scalarwave_spectral import spec

# test projection of 2D functions on grid
np.set_printoptions(precision=4)
x, y   = spec.chebnodes(28), spec.chebnodes(28)
xx, yy = np.meshgrid(x, y) 
zz     = np.exp(-(xx)**2.0/0.1) + np.exp(-(yy)**2.0/0.1)
f      = lambda x, y: np.exp(-(x)**2.0/0.1) + np.exp(-(y)**2.0/0.1)
print "- Computing analytic solution."

# project the analytic function
# extremeley restricted in the domain in which this works. FIX IT.
zzP    = spec.projectfunction2D(f, 28, spec.chebnodes(28))	

print "- Finished projecting analytic solution."
print np.mean(zz - zzP)

if(1):
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure()
	ax  = fig.add_subplot(111, projection='3d')
	ax.plot_wireframe(xx, yy, zz, linewidth=0.8)
	ax.plot_wireframe(xx, yy, zzP, color='k', linewidth=0.8)
	plt.show()