#==========================================================
# Utilities to solve the wave equaiton on a spectral grid
# Soham 2017
# Most functions are borrowed from someone else's codes. 
#==========================================================

import math
import numpy as np

# <https://github.com/UBC-Astrophysics/Spectral>
# CHEB  compute D = differentiation matrix, x = Chebyshev grid
def cheb(N):
  if N==0:
      D=0
      x=1
  else:
      x = np.cos(math.pi*np.arange(0,N+1)/N)
      c = np.concatenate(([2],np.ones(N-1),[2]))*(-1)**np.arange(0,N+1)
      X = np.tile(x,(N+1,1))
      dX = X-np.transpose(X)
      c=np.transpose(c)
      D=-np.reshape(np.kron(c,1/c),(N+1,N+1))/(dX+np.eye(N+1))
      D  = D - np.diagflat(np.sum(D,axis=1))	# diagonal entries
  return D,x

# <https://github.com/mikaem/spmpython>
# CLENCURT nodes x (Chebyshev points) and weights w for Clenshaw-Curtis
# quadrature
def clencurt(N):
    theta = np.pi*np.arange(0,N+1)/N
    x = np.cos(theta)
    w = np.zeros(N+1)
    ii = np.arange(1,N)
    v = np.ones(N-1)
    if np.mod(N,2)==0:
        w[0] = 1./(N**2-1)
        w[N] = w[0]
        for k in np.arange(1,int(N/2.)):
            v = v-2*np.cos(2*k*theta[ii])/(4*k**2-1)
        v = v - np.cos(N*theta[ii])/(N**2-1)
    else:
        w[0] = 1./N**2
        w[N] = w[0]
        for k in np.arange(1,int((N-1)/2.)+1):
            v = v-2*np.cos(2*k*theta[ii])/(4*k**2-1)
    w[ii] = 2.0*v/N
    return w

clencurt(5)

# function to plot the chebychev grid
def plotgrid(xx, tt):
  import matplotlib.pyplot as plt
  plt.plot(xx, tt, 'r-o')
  plt.plot(tt, xx, 'r-o')

  plt.plot(tt[0], xx[0], 'y-o')
  plt.plot(tt[-1], xx[-1], 'y-o')

  plt.plot(tt[:, 0], xx[:, 0], 'm-o')
  plt.plot(tt[:, -1], xx[:, -1], 'm-o')

  plt.ylim(-1.2, 1.2)
  plt.xlim(-1.2, 1.2)
  plt.xlabel(r"$x$")
  plt.ylabel(r"$t$")
  plt.savefig("../../output/chebgrid.png")
  plt.close()