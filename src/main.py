#==========================================================
# Code to solve the scalar wave equation using a 
# discretized lagrangian
#==========================================================

import numpy as np
import utilities as util
import matplotlib.pyplot as plt
from numpy import linalg as LA

N = 3

# Setting up a (N+1) x (N+1) points (t, x) cheb grid
D0, t = util.cheb(N) 
D1, x = util.cheb(N) 

if(0):	#plot grid
	xx,tt = np.meshgrid(t,x)
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

# construct the differentiation matrices
D20 = np.dot(D0,D0)
D21 = np.dot(D1,D1)
I   = np.eye(N+1)	

# construct the weight matrix
V = np.outer(util.clencurt(N), util.clencurt(N))
W = np.outer(np.ravel(V), np.ravel(V))

# construct the main operator
D = -np.kron(I,D20) + np.kron(D21,I)
A = np.dot(D, W)
A = A + np.transpose(A)

#impose boundary conditions
A = np.lib.pad(A, (0,4*(N+1) - 4), 'constant', constant_values=(0))

# FIXME: Very hacky way of setting the boundary values. Think of something smarter.
# loc = [0, 1, 2, 3, 4,    5,    6, 7, 8, 11, 12, 15]
# BC  = [0, 1, 1, 0, 0, 0.01, 0.01, 0, 0 ,0 , 0,   0]

loc = [0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15]
BC  = [0, 1, 1, 0, 0, 0, 0 ,0 , 0,  0,   0, 0]
row = 0

for _k in loc:
	A[row+16][_k] = 1
	A[_k][row+16] = 1
	row +=1

# find Eigenvalues (w) and Eigen vectors (v)
w, v = LA.eig(A)

if(0):
	plt.plot(w, 'ro')
	plt.axhline([0])
	plt.title("Eigen Values")
	plt.show()

if(0):
	plt.plot(v[:,2], '-')
	plt.title("Eigen Vectors")
	plt.show()	

#solve the system
b = np.zeros((N+1)**2.0)
b = np.hstack((b, BC))

u = np.linalg.solve(A, b)
u = u[:(N+1)**2]
uu = np.reshape(u, (N+1, N+1))

if(1):
	for t, i in enumerate(uu):
		plt.plot(i, label='t=%r'%t)
	plt.legend()
	plt.show()