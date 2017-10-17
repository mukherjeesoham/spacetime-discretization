#==========================================================
# Laplace eq. in 1D
# Soham 8 2017
#==========================================================

import numpy as np
import utilities as util
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import interpolate

#------------------------------------------------
# Grid
#------------------------------------------------

# Creates N+1 points.
N = 2

# Construct Chebyshev differentiation matrices.
D1, x = util.cheb(N) 

#------------------------------------------------
# Construct derivate + integral operators
#------------------------------------------------

D = np.dot(D1, D1)
W = np.diag(np.ravel(util.clencurt(N)))

# construct the main operator
A = np.dot(W, D)
# A = A + np.transpose(A)

# construct the vector
b  = np.zeros((N+1)+2)

# where do you want to impose the BCs?
BD = np.zeros(N+1)
BD[0] = 1
BD[-1] = 1

A  = np.lib.pad(A,  (0, 2), 'constant', constant_values=(0))
BC = np.where(BD==1)[0]

# set the A matrix
for _i, _j in enumerate(np.arange((N+1), (N+1) + len(BD[BD==1]))):
	A[BC[_i]][_j] = A[_j][BC[_i]] = 1

b = np.zeros(N+1 + len(BD[BD==1]))
b[-1] = 1.0
b[-2] = 0.0

#------------------------------------------------
# solve the system
#------------------------------------------------
u = np.linalg.solve(A, b)
u = u[:-2]
plt.plot(x,u, 'r-o', markersize=2, linewidth=0.6)
plt.show()
