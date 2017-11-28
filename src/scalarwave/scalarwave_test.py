from scalarwave_classes import patch, spec, multipatch
import numpy as np

if(1):	# FIXME: Broken while testing
	# test if the patch transformations are working as expected
	PATCH = patch(100, 2)
	MULTIPATCH = multipatch(2, 100)
	X00, Y00  = multipatch.gridtopatch(MULTIPATCH, PATCH, [0,0]) 
	X01, Y01  = multipatch.gridtopatch(MULTIPATCH, PATCH, [0,1]) 
	X10, Y10  = multipatch.gridtopatch(MULTIPATCH, PATCH, [1,0]) 
	X11, Y11  = multipatch.gridtopatch(MULTIPATCH, PATCH, [1,1]) 

	XF, YF    = spec.chebnodes(100), spec.chebnodes(200)

	def gridexp(X, Y):
		XX, YY  = np.meshgrid(X, Y)
		return XX, YY, np.exp(-XX**2.0/0.1 + -YY**2.0/0.1)

	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import axes3d
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_wireframe(gridexp(XF, YF)[0], gridexp(XF, YF)[1], gridexp(XF, YF)[2], linewidth=0.2)
	ax.plot_wireframe(gridexp(X00, Y00)[0], gridexp(X00, Y00)[1], gridexp(X00, Y00)[2], \
		rstride=10, cstride=10, color='r', linewidth=0.4)
	ax.plot_wireframe(gridexp(X11, Y11)[0], gridexp(X11, Y11)[1], gridexp(X11, Y11)[2], \
		rstride=10, cstride=10, color='g', linewidth=0.4)
	ax.plot_wireframe(gridexp(X10, Y10)[0], gridexp(X10, Y10)[1], gridexp(X10, Y10)[2], \
		rstride=10, cstride=10, color='m', linewidth=0.4)
	ax.plot_wireframe(gridexp(X01, Y01)[0], gridexp(X01, Y01)[1], gridexp(X01, Y01)[2], \
		rstride=10, cstride=10, color='y', linewidth=0.4)
	plt.show()


if(0):
	computationaldomain = multipatch(npatches=2, nmodes=40, \
							leftboundary  = lambda x: np.sin(np.pi*x*0),
							rightboundary = lambda x: np.sin(np.pi*x*0),
							potential = None)
	solution = computationaldomain.globalsolve()


#===============================================================
# Tests for scalar wave equation in Minkowski Spacetime
# Soham M 10/2017
#===============================================================

import numpy as np
from scalarwave_classes import patch, spec, conv


if(0):
	#------------------------------------------------------
	# create and solve for different boundary conditions 
	# on a single patch
	#------------------------------------------------------
	PATCH = patch(40)
	X 	  = patch.chebnodes(PATCH)
	print "Finished computing the nodes on the patch."
	OP	  = patch.operator(PATCH)
	print "Finished computing the operator"
	BC 	  = patch.setBCs(PATCH, fn = lambda X, Y: np.sin(X)*np.exp((-X**2.0)/0.1)*np.exp((-Y**2.0)/(0.1)))
	print "Finished computing BCs"
	VAL   = patch.solve(PATCH, BC, OP)
	print "Finished solving the linear equation"

	import matplotlib.pyplot as plt
	plt.imshow(VAL)
	plt.colorbar()
	plt.show()


if(1):
	#------------------------------------------------------
	# test p-convergence
	#------------------------------------------------------
	NP   = np.arange(2, 20, 2)
	# CONV = conv(NP, FN = lambda X, Y: np.sin(X)*np.exp((-X**2.0)/0.1)*np.exp((-Y**2.0)/(0.1)))
	CONV = conv(NP, BROWfn = lambda X : np.exp((-X**2.0)/0.1), 
					BCOLfn = lambda Y: np.exp((-Y**2.0)/(0.1)))
	conv.pconv(CONV, show=1)

if(0):
	#------------------------------------------------------
	# test L-inf error and where the error occurs
	#------------------------------------------------------
	PATCH = patch(20)
	X 	  = patch.chebnodes(PATCH)
	OP	  = patch.operator(PATCH)
	# BC 	  = patch.setBCs(PATCH, BROW = np.sin(np.pi*X), BCOL = np.sin(np.pi*X))
	BC 	  = patch.setBCs(PATCH, fn = lambda X, Y: np.sin(X)*np.exp((-X**2.0)/0.1)*np.exp((-Y**2.0)/(0.1)))
	SOL40   = patch.solve(PATCH, BC, OP)

	PATCH = patch(39)
	X 	  = patch.chebnodes(PATCH)
	OP	  = patch.operator(PATCH)
	# BC 	  = patch.setBCs(PATCH, BROW = np.sin(np.pi*X), BCOL = np.sin(np.pi*X))
	BC 	  = patch.setBCs(PATCH, fn = lambda X, Y: np.sin(X)*np.exp((-X**2.0)/0.1)*np.exp((-Y**2.0)/(0.1)))
	SOL39   = patch.solve(PATCH, BC, OP)
	SOLP40  = patch.projectpatch(PATCH, 40)

	import matplotlib.pyplot as plt
	if(1):
		# check where the L-inf error is the maximum
		print "L-inf : ", np.amax(SOLP40  - SOL40)
		plt.imshow(SOLP40  - SOL40)
		plt.colorbar()
		plt.show()
	else:
		# check where the solution is symmetric
		print "L-inf (symmetric): ", np.amax(SOL40 - SOL40.T)
		plt.imshow(SOL40 - SOL40.T)
		plt.colorbar()
		plt.show()


if(0):	
	# test transformation code
	PATCH  = patch(29, 2)

	X, Y   = patch.patchtogrid(PATCH, [0,0])
	XX, YY = np.meshgrid(X, Y)
	Z00 = np.exp(-np.pi*XX**2.0)*np.sin(np.pi*YY)

	X, Y   = patch.patchtogrid(PATCH, [0,1])
	XX, YY = np.meshgrid(X, Y)
	Z01 = np.exp(-np.pi*XX**2.0)*np.sin(np.pi*YY)

	X, Y   = patch.patchtogrid(PATCH, [1,0])
	XX, YY = np.meshgrid(X, Y)
	Z10 = np.exp(-np.pi*XX**2.0)*np.sin(np.pi*YY)

	X, Y   = patch.patchtogrid(PATCH, [1,1])
	XX, YY = np.meshgrid(X, Y)
	Z11 = np.exp(-np.pi*XX**2.0)*np.sin(np.pi*YY)

	RP = np.block([[Z00, Z01], [Z10, Z11]]) 

	MM, NN = np.meshgrid(spec.chebnodes(59), spec.chebnodes(59))
	FP = np.exp(-np.pi*MM**2.0)*np.sin(np.pi*NN)

	import matplotlib.pyplot as plt
	plt.subplot(1,2,1)
	plt.imshow(RP)
	plt.subplot(1,2,2)
	plt.imshow(FP)
	plt.show()
