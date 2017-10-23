import math
import numpy as np

def cheb(N):
  """
  see <https://github.com/UBC-Astrophysics/Spectral>
  CHEB compute D = differentiation matrix, x = Chebyshev grid
  """
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

def clencurt(N):
  """
  see <https://github.com/mikaem/spmpython>
  CLENCURT nodes x (Chebyshev points) and weights w 
  for Clenshaw-Curtis quadrature
  """
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

def operator(N):
  Du, u  = cheb(N)
  Dv, v  = cheb(N)
  uu, vv = np.meshgrid(u,v)
  I  = np.eye(N+1)
  DU = np.kron(Du, I)
  DV = np.kron(I, Dv)
  D  = np.dot(DU,DV) + np.dot(DV,DU)      # operator
  V  = np.outer(clencurt(N), clencurt(N))
  W  = np.diag(np.ravel(V))                # integration weights
  A  = W.dot(D)
  BC = np.zeros((N+1,N+1))
  BC[0, :] = BC[:, 0]  = 1  
  A[np.where(np.ravel(BC)==1)[0]] = np.eye((N+1)**2)[np.where(np.ravel(BC)==1)[0]]  
  return A

def makeboundaryvec(N, bcol, brow):
  b = np.eye(N+1)*0.0
  b[:,  0] = bcol
  b[0,  :] = brow
  return np.ravel(b)

def makeinitialdata(x):
  return np.sin(np.pi*x)

def setzero(x):
  return np.zeros(len(x))

def makeglobalgrid(M):
  grid = np.zeros((M,M))
  for index, val in np.ndenumerate(grid):
    grid[index] = np.sum(index)
  return grid
