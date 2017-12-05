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

if(0):
	"""
	Test multipatch solution without futures
	"""
	computationaldomain = multipatch(npatches=1, nmodes=50, \
							leftboundary  = lambda x: np.exp(-x**2.0/0.1), \
							rightboundary = lambda y: np.exp(-y**2.0/0.1), \
							potential 	  = None)
	domain 	 = computationaldomain.globalsolve()
	solution = computationaldomain.assemblegrid(domain) 
	patch.plotpatch(solution)

if(1):
	"""
	Test BCs
	"""
	computationaldomain = multipatch(npatches=1, nmodes=60, \
							leftboundary  = lambda x: np.exp(-x**2.0/0.1), \
							rightboundary = lambda y: np.exp(-y**2.0/0.1), \
							potential 	  = None)
	X, BC   	 = computationaldomain.testglobalsolve()
	plt.plot(X, BC[:, 0], 'k.')
	plt.plot(X, np.exp(-X**2.0/0.1), 'm--', linewidth=0.4)
	plt.grid()
	plt.xticks([])
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
	nmodes  = 50
	f = lambda x: np.exp(-x**2.0/0.1)
	
	x  = spec.chebnodes(npoints)
	v  = spec.projectfunction1D(f, nmodes, npoints)
	s  = f(x)
	print "L2 Error norm: ", np.sqrt(np.abs(np.dot(spec.chebweights(npoints), \
						(v-s)**2.0)/np.abs(np.dot(spec.chebweights(npoints), s**2.0))))
	plt.plot(x, v, 'k.')
	plt.plot(x, f(x), 'm--', linewidth=0.5)
	plt.show()

if(0):	# test 2D functions FIXME
	npoints = 9
	nmodes  = 9
	f = lambda x, y: np.exp(-x**2.0/0.1) + np.exp(-y**2.0/0.1)
	x = spec.chebnodes(npoints)
	xx, yy = np.meshgrid(x, x)
	s  = f(xx, yy)
	v  = spec.projectfunction2D(f, nmodes, npoints)
	
if(0):
	npoints = 20
	x  = spec.chebnodes(npoints)
	w  = spec.chebweights(npoints)
	s  = np.insert(np.cumsum(w) - 1, 0, -1)
	xx, yy = np.meshgrid(spec.chebnodes(npoints), spec.chebnodes(npoints))
	z = np.exp(-xx**2.0) + np.exp(-yy**2.0)
	
	# FIXME The normalization doens't work for every case
	# z = np.sin(np.pi*xx) + np.sin(np.pi*yy)

	from matplotlib import cm
	import matplotlib.patches as patches
	znorm = z/np.amax(z)
	colors = cm.viridis(znorm)
	fig = plt.figure()
	ax  = fig.add_subplot(111, aspect='equal')
	ax.set_xlim([-1, 1])
	ax.set_ylim([-1, 1])
	ax.plot(xx, yy, 'k.', markersize=2)
	for i in range(len(s)-1):
		for j in range(len(s)-1):
			ax.add_patch(patches.Rectangle((s[i], s[j]), s[i+1] - s[i], s[j+1] - s[j], fill=True, facecolor=colors[i,j]))
			ax.add_patch(patches.Rectangle((s[i], s[j]), s[i+1] - s[i], s[j+1] - s[j], fill=False, linewidth=0.2))
	plt.show()