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
  m = 2
  Du, u  = cheb(N)
  Dv, v  = cheb(N)
  uu, vv = np.meshgrid(u,v)
  I  = np.eye(N+1)
  DU = np.kron(Du, I)
  DV = np.kron(I, Dv)

  # add a forced potential
  D  = np.dot(DU,DV) + np.dot(DV,DU)
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

def makeglobalgrid(M):
  grid = np.zeros((M,M))
  for index, val in np.ndenumerate(grid):
    grid[index] = np.sum(index)
  return grid

def makeglobalchart(M,N):
  b = np.zeros((M*N + 1, M*N + 1))
  Du, u  = cheb(N*M)
  Dv, v  = cheb(N*M)
  for index, value in np.ndenumerate(b):
    t = (u[index[0]] + v[index[1]])/2.0
    r = (v[index[0]] - u[index[1]])/2.0
    b[index] = np.cos(4*np.pi*t)*np.exp(-t**2.0)*np.exp(-r**2.0/(0.01))
  
  columns = np.linspace(0, M*(N+1), M+1)[1:-1]
  for column in columns:
    b = np.insert(b, int(column), b[int(column), :], 0) 
    b = np.insert(b, int(column), b[:, int(column)], 1) 

  pieces = np.zeros((M,M,N+1,N+1))
  hsplit = np.hsplit(b, M)
  for m, element in enumerate(hsplit):
    vsplit = np.vsplit(element, M)
    for n, chunk in enumerate(vsplit):
      pieces[m,n] = chunk
  return pieces