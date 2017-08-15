#==========================================================
# Code to solve the scalar wave equation using a 
# discretized lagrangian
# Soham 2017
#==========================================================

import numpy as np
import utilities as util
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import interpolate

# FIXME: > Singular matrix with projection operators
#		 > For very low resolution the solver works
#		 > for Poisson eqn. However for wave equation, 
#		 > things don't work equally well.
			
#------------------------------------------------
# Grid
#------------------------------------------------

# Creates N+1 points.
N = 3

# Construct Chebyshev differentiation matrices.
D0, t = util.cheb(N) 
D1, x = util.cheb(N) 
xx,tt = np.meshgrid(t,x)

if(0):	#plot grid

	plt.plot(xx, tt, 'r-o')
	plt.plot(tt, xx, 'r-o')
	
	plt.plot(tt[0], xx[0], 'g-o')
	plt.plot(tt[-1], xx[-1], 'g-o')

	plt.plot(tt[:, 0], xx[:, 0], 'm-o')
	plt.plot(tt[:, -1], xx[:, -1], 'm-o')

	plt.ylim(-1.2, 1.2)
	plt.xlim(-1.2, 1.2)
	plt.xlabel(r"$x$")
	plt.ylabel(r"$t$")
	plt.show()

#------------------------------------------------
# Construct operators
#------------------------------------------------



# Construct the derivative operator (acting on u)
I   = np.eye(N+1)	
D   = np.kron(I,np.dot(D0, D0)) + np.kron(np.dot(D1, D1), I)

# construct the weight matrix
V = np.outer(util.clencurt(N), util.clencurt(N))
W = np.diag(np.ravel(V)) + np.diag(np.ravel(V))

# construct the integral + operator
A = D.dot(W)
A = A + np.transpose(A)

# construct projection operators
B = np.zeros((N+1,N+1))

#first derivative
B[0] =  1
BN = np.dot(np.diag(np.ravel(B)), np.kron(I,D0))	

#bordering points
B[:, 0] = B[:, -1]= B[0] = B[-1] = 1
BD = np.diag(np.ravel(B))

#------------------------------------------------
# Solve
#------------------------------------------------

if(1):	# in case you want to use Lagrange multipliers 
	A  = np.lib.pad(A, (0,4*(N+1)-4), 'constant', constant_values=(0))
	BC = np.where(np.diag(BD) > 0)[0]
	for _i, _j in enumerate(np.arange((N+1)**2, (N+1)**2 + 4*(N+1)-4)):
		A[BC[_i]][_j] = A[_j][BC[_i]] = 1
	
	# construct b
	uu = np.zeros((N+1, N+1))

	#set the boundary conditions here.
	uu[0] = uu[-1] = 0
	uu[:, 0] = uu[:, -1] = 10
	
	b = np.ravel(uu)[np.where(np.diag(BD) > 0)[0]]
	b = np.append(np.zeros((N+1)**2), b)

	# solve the system
	print("Solving a dimension %r linear system..."%np.shape(A)[0])
	u = np.linalg.solve(A, b)
	uu = np.reshape(u[:(N+1)**2.0], (N+1, N+1))
	print("Completed solve.")

else:	# FIXME: Singular Matrix
	A = A.dot(BD).dot(BN)	#dot with the differential operators you need.
	
	uu = np.zeros((N+1, N+1))
	#set the boundary conditions here.
	uu[0] = uu[-1] = 0
	uu[:, 0] = uu[:, -1] = 10	
	b = np.ravel(uu)
	
	# solve the system
	print("Solving a dimension %r linear system..."%np.shape(A)[0])
	u = np.linalg.solve(A, b)
	uu = np.reshape(u, (N+1, N+1))
	print("Completed solve.")


#------------------------------------------------
#analysis
#------------------------------------------------
if(0):	# find Eigenvalues (w) and Eigen vectors (v)
	w, v = LA.eig(A)
	if(1):
		plt.semilogy(w, 'ro')
		plt.axhline([0])
		plt.title("Eigen Values")
		plt.show()

	if(0):
		plt.plot(v[:,2], '-')
		plt.title("Eigen Vectors")
		plt.show()	

if(0):	#interpolate or not?
	print("Interpolating grid.")
	f = interpolate.interp2d(x, t, uu, kind='cubic')
	xnew = np.linspace(-1, 1, 1e+2)
	ynew = np.linspace(-1, 1, 1e+2)
	Z = f(xnew, ynew)
	X, Y = np.meshgrid(xnew, ynew)
else:
	Z = uu
	X, Y = np.meshgrid(t, x)

if(1):	#plot
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
	plt.show()