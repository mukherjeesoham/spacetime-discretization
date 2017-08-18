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

#------------------------------------------------
# Grid
#------------------------------------------------

# Creates N+1 points.
N = 3

# Construct Chebyshev differentiation matrices.
D0, t = util.cheb(N) 
D1, x = util.cheb(N) 
xx,tt = np.meshgrid(t,x)

if(1):
	util.plotgrid(xx, tt)

#------------------------------------------------
# Construct derivate + integral operators
#------------------------------------------------

# Construct the derivative operator
I =  np.eye(N+1)	
D = np.kron(I,np.dot(D0, D0)) + np.kron(np.dot(D1, D1), I)

# construct the weight matrix [XXX: Check]
V = np.outer(util.clencurt(N), util.clencurt(N))
W = np.diag(np.ravel(V))

# construct the integral + operator
A = D.dot(W)
A = A + np.transpose(A)

#------------------------------------------------
# Construct templates for boundary operators
#------------------------------------------------

# create template for the projection operators
loc_BD = np.zeros((N+1,N+1))

# choose the boundary points on three sides of the grid.
loc_BD[:, 0] = loc_BD[:, -1]= loc_BD[0] = loc_BD[-1] = 1
BD = np.diag(np.ravel(loc_BD))

#------------------------------------------------
# Impose boundary conditions
#------------------------------------------------

# imposing the Dirichlet BCs using Lagrange multipliers
A  = np.lib.pad(A,  (0, len(BD[BD==1])), 'constant', constant_values=(0))

# construct the vector to be used 
# for imposing the boudanry condition
BC = np.where(np.ravel(loc_BD)==1)[0]
for _i, _j in enumerate(np.arange((N+1)**2, (N+1)**2 + len(BD[BD==1]))):
	A[BC[_i]][_j] = A[_j][BC[_i]] = 1

# construct the final matrix
F  = A

# construct the vector to be used 
# for imposing the boudanry condition
b  = np.zeros((N+1)**2.0)

# set the Dirichlet boundary values
uu = np.zeros((N+1, N+1))
uu[:, 0]  = 1
uu[:, -1] = 1
uu[0]     = 0
uu[-1]    = 0

u = np.ravel(uu)
u = u[np.where(np.ravel(loc_BD)==1)[0]]
b = np.append(b, u)

#------------------------------------------------
# solve the system
#------------------------------------------------

print("Solving a dimension %r linear system..."%np.shape(F)[0])
if np.linalg.det(F) != 0:	
	u = np.linalg.solve(F, b)
	uu = np.reshape(u[:(N+1)**2.0], (N+1, N+1))
	print("Completed solve.")
else:
	print "Encountered Singular Matrix. Aborting solver."

#------------------------------------------------
#analysis
#------------------------------------------------
if(1):	# find Eigenvalues (w) and Eigen vectors (v)
	print "\nAnalysing the F matrix..."
	w, v = np.linalg.eig(F)

	plt.semilogy(w, 'ro')
	plt.axhline([0])
	plt.title("Eigen values for a %r-dimensional matrix"%(np.shape(F)[0]))
	plt.savefig("../output/laplace_eigen_values.png")
	plt.close()

if(1):	#plot the solution
	if(1):	#interpolate or not?
		print("Interpolating grid.")
		f = interpolate.interp2d(x, t, uu, kind='cubic')
		xnew = np.linspace(-1, 1, 40)
		ynew = np.linspace(-1, 1, 40)
		Z = f(xnew, ynew)
		X, Y = np.meshgrid(xnew, ynew)
	else:
		Z = uu
		X, Y = np.meshgrid(t, x)

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	plt.xlabel("x")
	plt.ylabel("t")
	ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)


	plt.savefig("../output/laplace_solution.png")
	plt.close()


