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

start_main = time.time()

# Creates N+1 points.
N = 10

# Construct Chebyshev differentiation matrices.
D0, t = util.cheb(N)
D1, x = util.cheb(N)
D2, y = util.cheb(N)
D3, z = util.cheb(N)

D02 = np.dot(D0, D0)
D12 = np.dot(D1, D1)
D22 = np.dot(D2, D2)
D32 = np.dot(D3, D3)

#------------------------------------------------
# Construct derivate + integral operators
#------------------------------------------------

# Construct the derivative operator
I = np.eye(N+1)

DD = - np.kron(I, np.kron(I, np.kron(I, D02))) \
	 + np.kron(I, np.kron(np.kron(I, D12), I)) \
 	 + np.kron(np.kron(np.kron(I, D22), I), I) \
 	 + np.kron(np.kron(np.kron(D32, I), I), I)

Dt = np.kron(I, np.kron(I, np.kron(I, D0)))
Dx = np.kron(I, np.kron(np.kron(I, D12), I))
Dy = np.kron(np.kron(np.kron(I, D22), I), I)
Dz = np.kron(np.kron(np.kron(D32, I), I), I)

# construct the weight matrix
V = np.outer(np.outer(np.outer(util.clencurt(N), util.clencurt(N)), util.clencurt(N)), util.clencurt(N))
W = np.diag(np.ravel(V))

# construct the main operator
A = W.dot(DD)

#------------------------------------------------
# Construct templates for boundary operators
#------------------------------------------------

# Choose the boundary points on
# all sides of the grid and flatten the array out.
loc_BC = np.zeros((N+1, N+1, N+1, N+1))

loc_BC[0,  :, :, :]     =  1	#  initial time
loc_BC[-1, :, :, :]     = -1	#  final time

# spatial boundaries
loc_BC[:,  0, :,  :]     = 1	# x(L)	
loc_BC[:, -1, :,  :]     = 1	# x(R)
loc_BC[: , :,  0, :]     = 1	# y(L)
loc_BC[:,  :, -1, :]     = 1	# y(R)
loc_BC[:, : , :,  0]     = 1	# z(L)
loc_BC[:, :,  :, -1]     = 1	# z(R)

BC = np.ravel(loc_BC)

#------------------------------------------------
# Impose boundary conditions [using Lagrange multipliers]
#------------------------------------------------

def dirichlet(A, index):
	A[index] = np.zeros(np.shape(A)[0])
	A[index][index] = 1.0
	return A

def neumann(A, index):
	A[index] = Dt[index]
	return A

# reset matrix A
for _i,_r in enumerate(np.where(BC!=0)[0]):
	if BC[_r] == 1:		#impose Dirichlet BCs
		dirichlet(A, _r)
	elif BC[_r] == -1:	#impose Neumann BCs
		neumann(A, _r)

# set the Dirichlet boundary values
uu = np.zeros((N+1, N+1, N+1, N+1))

uu[0,  :, :, :]     = np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)	#  initial time
uu[-1, :, :, :]     = 0	#  final time

# spatial boundaries
uu[:,  0, :,  :]     = 0	# x(L)	
uu[:, -1, :,  :]     = 0	# x(R)
uu[: , :,  0, :]     = 0	# y(L)
uu[:,  :, -1, :]     = 0	# y(R)
uu[:, : , :,  0]     = 0	# z(L)
uu[:, :,  :, -1]     = 0	# z(R)

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
uu = np.reshape(u, (N+1, N+1, N+1, N+1))


#------------------------------------------------
#analysis
#------------------------------------------------
if(0):	# find Eigenvalues (w) and Eigen vectors (v)
	print "\nAnalysing the matrix..."
	w, v = np.linalg.eig(A)

end_main = time.time()
print "Time taken by code: %rs" %(end_main - start_main)/60.0
