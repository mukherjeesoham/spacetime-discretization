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
N = 3

# Construct Chebyshev differentiation matrices.
D1, x = util.cheb(N) 

#------------------------------------------------
# Construct derivate + integral operators
#------------------------------------------------

D = np.dot(D1, D1)
W = np.diag(np.ravel(util.clencurt(N)))

# construct the main operator
A = np.dot(W, D)

# construct the vector
b  = np.zeros(N+1)
b[0]  = 0.0
b[-1] = 1.0

A[0] = A[-1] = np.zeros(4)
A[0][0] = 1
A[-1][-1] = 1

print A

#------------------------------------------------
# solve the system
#------------------------------------------------
u = np.linalg.solve(A, b)
u = u
plt.plot(x,u, 'b-o', markersize=2, linewidth=0.6)
plt.show()
