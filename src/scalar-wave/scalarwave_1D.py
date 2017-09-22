#==========================================================
# Code to solve the scalar wave equation using a
# discretized lagrangian
# Soham 8 2017
#==========================================================

import numpy as np
import utilities as util
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import interpolate
import time

#------------------------------------------------
# Grid
#------------------------------------------------

# Creates N+1 points.
N = 20

# Construct Chebyshev differentiation matrices.
D0, t = util.cheb(N)
D1, x = util.cheb(N)
tt, xx = np.meshgrid(t,x)

if(1):
	util.plotgrid(tt, xx)

#------------------------------------------------
# Construct derivate + integral operators
#------------------------------------------------

# Construct the derivative operator
I = np.eye(N+1)
D = - np.kron(I,np.dot(D0, D0)) + np.kron(np.dot(D1, D1), I)
Dt = np.kron(I, D0)
Dx = np.kron(D1, I)

print np.shape(D0)
# construct the weight matrix
V = np.outer(util.clencurt(N), util.clencurt(N))
W = np.diag(np.ravel(V))

# construct the main operator
A = W.dot(D)

#------------------------------------------------
# Construct templates for boundary operators
#------------------------------------------------

# Choose the boundary points on
# three sides of the grid and flatten the array out.
loc_BC = np.zeros((N+1,N+1))
loc_BC[0]     = -1	#  (x,  1) final time
loc_BC[-1]    = 1	#  (x, -1) intial time
loc_BC[:, 0]  = 1	# ( 1,  y) left boundary 
loc_BC[:, -1] = 1	# (-1,  y) right boundary

BC = np.ravel(loc_BC)

#------------------------------------------------
# Impose boundary conditions [using Lagrange multipliers]
#------------------------------------------------

def dirichlet(A, index):
	A[index] = np.zeros(np.shape(A)[0])
	A[index][index] = 1.0
	return A

def neumann(A, index):
	A[index] = Dx[index]
	return A

# reset matrix A
for _i,_r in enumerate(np.where(BC!=0)[0]):
	if BC[_r] == 1:		#impose Dirichlet BCs
		dirichlet(A, _r)
	elif BC[_r] == -1:	#impose Neumann BCs
		neumann(A, _r)

# construct the vector to be used
# for imposing the boudanry condition
b  = np.zeros((N+1)**2)

# set the Dirichlet boundary values
uu = np.zeros((N+1, N+1))
uu[:, 0]  = 0.0
uu[:, -1] = 0.0
uu[-1]    = - np.sin(np.pi*x) #2.0 #-(x**2.0 - 1) #
uu[0]     = 0.0

uu_boudary = np.copy(uu)
b = np.ravel(uu)

#------------------------------------------------
# solve the system
#------------------------------------------------

print("Solving a dimension %r linear system..."%np.shape(A)[0])
start = time.time()
u = np.linalg.solve(A, b)
end = time.time()
print "Time taken by solver: %rs" %(end - start)
uu = np.reshape(u[:(N+1)**2], (N+1, N+1))

#------------------------------------------------
#analysis
#------------------------------------------------
if(0):	# find Eigenvalues (w) and Eigen vectors (v)
	print "\nAnalysing the F matrix..."
	w, v = np.linalg.eig(A)

	plt.semilogy(w, 'ro')
	plt.axhline([0])
	plt.title("Eigen values for a %r-dimensional matrix"%(np.shape(A)[0]))
	plt.savefig("../output/laplace_eigen_values.png")
	plt.close()


if(1):	#plot the solution
	if(0):	#interpolate or not?
		print("Interpolating grid.")
		f    = interpolate.interp2d(t, x, uu, kind='cubic')
		xnew = np.linspace(-1, 1, 30)
		ynew = np.linspace(-1, 1, 30)
		Z    = f(xnew, ynew)
		X, Y = np.meshgrid(xnew, ynew)
		# S    = ls.simple_BCs(X, Y)
		# Err  = np.sqrt(np.mean((Z - S)**2.0))
		# print "N = %r \t Error: %r "%(N+1, Err)
	else:
		Z    = uu
		X, Y = np.meshgrid(t, x)
		# S    = ls.simple_BCs(X, Y)
		# Err  = np.sqrt(np.mean((Z - S)**2.0))
		# print "N = %r \t Error: %r "%(N+1, Err)

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	plt.xlabel("x")
	plt.ylabel("t")
	if(0):	# BCs
		ax.plot_wireframe(xx, yy, uu_boudary, rstride=1, cstride=1)
	elif(0):	# analytical solution
		ax.plot_wireframe(X, Y, S, rstride=1, cstride=1, color='r', linewidth=0.4)
	else:		# computed solution
		ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='m', linewidth=0.6)
	plt.show()
