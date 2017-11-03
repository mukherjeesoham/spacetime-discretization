#==========================================================
# Code to solve the scalar wave equation using the
# discretized action in Cartesian Coordinates.
# Soham 8 2017
#==========================================================

import numpy as np
import sw_utilities as util
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import matplotlib 

#------------------------------------------------
# Grid
#------------------------------------------------

def main(N):					# Creates N+1 points.	
	start = time.time()
	# Construct Chebyshev differentiation matrices.
	D0, t  = util.cheb(N)
	D1, x  = util.cheb(N)
	tt, xx = np.meshgrid(t,x)

	#------------------------------------------------
	# Construct derivate + integral operators
	#------------------------------------------------

	# Construct the derivative operator
	I  = np.eye(N+1)

	# XXX: How to fix the orientation of the Kronecker sum?
	D  = - np.kron(np.dot(D0, D0), I) + np.kron(I,np.dot(D1, D1)) 
	Dx = np.kron(I, D0)
	Dt = np.kron(D1, I)

	# construct the weight matrix
	V = np.outer(util.clencurt(N), util.clencurt(N))
	W = np.diag(np.ravel(V))

	# construct the main operator
	A = W.dot(D)

	#------------------------------------------------
	# Impose boundary conditions and construct b
	#------------------------------------------------

	BC = np.zeros((N+1,N+1))

	# Dirichlet boundary conditions
	BC[0, :] = BC[:, 0] = BC[:, -1] = BC[-1, :] = 1	
	A[np.where(np.ravel(BC)==1)[0]] = np.eye((N+1)**2)[np.where(np.ravel(BC)==1)[0]]

	# Neumann boundary conditions 
	BC[0, :] =  -1
	A[np.where(np.ravel(BC)==-1)[0] - (N + 1)] = Dt[np.where(np.ravel(BC)==-1)[0]]

	# set the Dirichlet boundary values
	b = np.zeros((N+1, N+1))

	BC = "Sine"

	if BC == "Gaussian":
		b[:,  0] = b[:, -1] = np.exp(-10.0)	  # left & right boundaries
		b[0,  :] = np.exp(-10.0*x**2.0)    	# initial data	
	if BC == "Sine":
		b[:,  0] = b[:, -1] = 0.0	  # left & right boundaries
		b[0,  :] = -np.sin(np.pi*x) # initial data	

	b[-1, :] = 0.0				  # time derivative
	b = np.ravel(b)

	#------------------------------------------------
	# solve the system
	#------------------------------------------------

	print("Solving a dimension %r linear system..."%np.shape(A)[0])
	u = np.linalg.solve(A, b)
	uu = np.reshape(u[:(N+1)**2], (N+1, N+1))

	#------------------------------------------------
	# Analysis
	#------------------------------------------------

	print "\nIs A symmetric?", np.allclose(A, A.T)

	def is_pos_def(x):
	    return np.all(np.linalg.eigvals(x) > 0)
	if(0):
		print "Is A positive definite?", is_pos_def(A)

	# FIXME: Check why you need the transpose.
	ss = np.transpose(np.sin(np.pi*xx)*np.cos(np.pi*tt))
	ee = ss - uu
	L2 =  np.sqrt(np.trace(np.abs(W*np.ravel(ee**2.0))))
	print "L2 error norm: ", L2

	end = time.time()
	runtime = end - start
	return L2, uu, xx, tt, ss, runtime


#===================================================
# Call function
#===================================================


if(1):	# single solution
	L2, uu, xx, tt, ss, runtime =main(20)

	if(1):
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		plt.xlabel(r"$x$")
		plt.ylabel(r"$t$")

		Z = (uu-uu.min())/(uu.max()-uu.min())
		colors = cm.viridis_r(Z)
		surf = ax.plot_surface(tt, xx, uu, rstride=1, cstride=1,
	               facecolors = colors, shade=False, linewidth=0.6)
		surf.set_facecolor((0,0,0,0))
		plt.show()

if(0):	# convergence
	N   = [3, 7, 15]
	Err  = np.zeros(len(N))
	Time = np.zeros(len(N))

	for _i, _n in enumerate(N):
		L2, uu, xx, tt, ss, runtime = main(_n)
		Err[_i]  = L2
		Time[_i] = runtime

	if(1):
		plt.semilogy(N, Err, 'm-o')
		plt.xlabel("N")
		plt.ylabel(r"$L_2$" + " norm")
	else:
		plt.plot(N, Time, 'm-o')
		plt.xlabel("N")
		plt.ylabel("Run time (s)")
	plt.show()










