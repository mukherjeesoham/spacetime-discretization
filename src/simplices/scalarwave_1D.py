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
N = 4

# Construct Chebyshev differentiation matrices.
D0, x = util.cheb(N)
D1, y = util.cheb(N)

# the standard tensor product grid
xx, yy = np.meshgrid(x,y)

fig, ax = plt.subplots(nrows=1, ncols=2)


ax[0].plot(xx,yy, 'o')


# TSEM grid

u = ((1 + xx)*(1-yy)/4.0)
v = (1 + yy)/2.0


ax[1].plot(u, v, 'o')
plt.show()
