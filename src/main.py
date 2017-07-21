#==========================================================
# Code to solve the scalar wave equation using a 
# discretized lagrangian
#==========================================================

import numpy as np
import cheb as cb
import matplotlib.pyplot as plt
from numpy import linalg as LA

N = 10

# Setting up a NxN (t, x) cheb grid
D0, t = cb.cheb(N) 
D1, x = cb.cheb(N) 
xx,tt = np.meshgrid(t,x)

if(0):	#plot grid
	plt.plot(xx, tt, 'g')
	plt.plot(tt, xx, 'm')
	plt.xlabel(r"$x$")
	plt.ylabel(r"$t$")
	plt.show()

# Now construct the differentiation matrices
# XXX: We do not take only the interior points since 
# we'll impose constraints there. 

D20 = np.dot(D0,D0)
D21 = np.dot(D1,D1)
I   = np.eye(N+1)	

# construct the main derivative operator
D = -np.kron(I,D20) + np.kron(D21,I)

#construc the integration matrix using trapezoidal rule
dt = np.ediff1d(t)
dx = np.ediff1d(x)
W  = 4*np.ones(np.shape(D))

# set all edge values
W[:,0] = W[:,-1] = W[0,:] = W[-1,:] = 2
# set all corner values
W[0,0] = W[0,-1] =  W[-1,0] = W[-1,-1] = 1

# FIXME: Note that the grid is no longer uniform. 
if(0):
	for index, val in np.ndenumerate(W):
		W[index] = val*dt[index[0]]*dx[index[1]]

# construct the final matrix C
A = np.dot(W, D)
F = np.transpose(A) + A

# find Eigenvalues (w) and Eigen vectors (v)
# XXX: These are complex quantities. What to visualize?
w, v = LA.eig(F)

if(0):
	plt.plot(w, 'ro')
	plt.title("Eigen Values")
	plt.show()

if(0):
	plt.plot(v, '-')
	plt.title("Eigen Vectors")
	plt.show()	

# solve the system of equations [See how to solve homogenous system
# of equations. lingalg.solve won't work. We might need to find the null space vectors.]
# np.linalg.solve(F)

# Use a vector b to constraint the solution in the first two time steps
u = np.zeros((len(t),len(x)))

# impose the boundary conditions
u[:, 0]  = +np.exp((x-0.0001)**2.0/10.25)
u[:, 1]  = +np.exp((x-0.0001)**2.0/10.25)
u[0, :]  = -1.2
u[-1, :] = -1.2

uu = np.ravel(u)

# now solve with linalg [Unfortunately this doesn't work either]
# uu = np.linalg.solve(F, uu)
