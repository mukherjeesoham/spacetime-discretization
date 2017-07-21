#==========================================================
# The function is taken as is from UBC-Astriphysics Git repo
# <https://github.com/UBC-Astrophysics/Spectral>
# which has Python implementations of Trethen's problems.
#==========================================================

import math
import numpy as np
# CHEB  compute D = differentiation matrix, x = Chebyshev grid

def cheb(N):
  if N==0:
      D=0
      x=1
  else:
      x = np.cos(math.pi*np.arange(0,N+1)/N)
      c = np.concatenate(([2],np.ones(N-1),[2]))*(-1)**np.arange(0,N+1)
#      X = np.transpose(np.tile(x,(N+1,1)))
      X = np.tile(x,(N+1,1))
      dX = X-np.transpose(X)
      c=np.transpose(c)
      D=-np.reshape(np.kron(c,1/c),(N+1,N+1))/(dX+np.eye(N+1))
#      D  = np.transpose(np.multiply(np.transpose(c),1/c)/
#                        (np.transpose(dX)+np.eye(N+1)))
      D  = D - np.diagflat(np.sum(D,axis=1))                 # diagonal entries

  return D,x
