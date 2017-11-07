#===============================================================
# Scalar wave equation in Minkowski Spacetime [Utilities]
# Soham M 10/2017
#===============================================================

import math
import numpy as np
import matplotlib 
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

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
  start = time.time()
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
  end = time.time()
  runtime = end - start
  print "Computed operator in %1.3fs" %(runtime) 
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

def makeglobalchartcopy(M,N):
  # FIXME: This is inconsistent
  b = np.zeros((M*N + 1, M*N + 1))
  Du, u  = cheb(N*M)
  Dv, v  = cheb(N*M)
  for index, value in np.ndenumerate(b):
    t = (u[index[0]] + v[index[1]])/2.0
    r = (v[index[0]] - u[index[1]])/2.0
    b[index] = np.sin(t)*np.exp((-t**2.0)/0.1)*np.exp((-r**2.0)/(0.1)) 
    # b[index] = np.cos(np.pi*t)*np.exp(-t**2.0)*np.exp(-r**2.0/(0.01))
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

def makeglobalchart(M,N):
  U = np.array([])
  V = np.array([])
  for patch in range(M):
    u = cheb(N)[1]
    v = cheb(N)[1]
    U = np.append(U, u + patch)
    V = np.append(V, v + patch)
  X = np.sort((2.0/(np.max(U) - np.min(U)))*(U - (np.max(U) + np.min(U))/2.0))
  Y = np.sort((2.0/(np.max(V) - np.min(V)))*(V - (np.max(V) + np.min(V))/2.0))
  b = np.zeros((M*(N + 1), M*(N + 1)))
  XX, YY = np.meshgrid(X,Y)
  for index, value in np.ndenumerate(b):
    t = (X[index[0]] + Y[index[1]])/2.0
    r = (X[index[0]] - Y[index[1]])/2.0
    b[index] = np.sin(t)*np.exp((-t**2.0)/0.1)*np.exp((-r**2.0)/(0.1)) 

  pieces = np.zeros((M,M,N+1,N+1))
  hsplit = np.hsplit(b, M)
  for m, element in enumerate(hsplit):
    vsplit = np.vsplit(element, M)
    for n, chunk in enumerate(vsplit):
      pieces[m,n] = chunk
  return pieces   

def assemblegrid(M, N, domain):
  I = []
  domain[M-1, M-1]
  for i in range(M):
    J = []
    for j in range(M):  
      J.append(domain[i,j])
    I.append(J)
  
  block = np.block(I)
  columns = np.linspace(0, M*(N+1), M+1)[1:-1]
  block = np.delete(block, columns, 0) 
  block = np.delete(block, columns, 1) 
  return block

def plotgrid(dictionary):
  domain = dictionary["domain"]
  M = dictionary["numpatches"]
  N = dictionary["size"]

  plt.imshow(domain)
  for k in range(1, M):
    plt.axvline([k*(N)], color='w')
    plt.axhline([k*(N)], color='w')
  plt.xlabel("u [top]")
  plt.ylabel("v")
  plt.axis("off")
  plt.colorbar()
  plt.savefig("./scalarwave_grid.png")
  plt.close()
  return None

def plotgrid3D(dictionary):
    domain = dictionary["domain"]
    M = dictionary["numpatches"]
    N = dictionary["size"]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.xlabel(r"$u$")
    plt.ylabel(r"$v$")

    if M==1:
      uu, vv = np.meshgrid(dictionary["chebnodes"], dictionary["chebnodes"])
    else:
      uu, vv = np.meshgrid(cheb(M*N)[1], cheb(M*N)[1])

    Z = (domain-domain.min())/(domain.max()-domain.min())
    colors = cm.viridis(Z)
    surf = ax.plot_surface(uu, vv, domain, rstride=1, cstride=1,
                 facecolors = colors, shade=False, linewidth=0.6)
    surf.set_facecolor((0,0,0,0))
    plt.show()

def eigenval(dictionary):
  A = dictionary["operator"]
  eigenvalues = np.linalg.eigvals(A)
  emax = np.amax(np.abs(eigenvalues))
  emin = np.amin(np.abs(eigenvalues))
  println()
  print "==> Eigenval (max/min): ", emax/emin
  println()
  return None
  
def println():
  print 40*'-'
  return None

#--------------------------------------------------------
# Test convergence
#--------------------------------------------------------

def extractcoeffs(domain):
  N  = np.shape(domain)[0] 
  x  = cheb(N-1)[1]
  CP = np.polynomial.chebyshev.chebval(x, np.eye(N))
  CM = np.kron(CP, CP)  
  return np.linalg.solve(CM, np.ravel(domain))

def extractvalues(coefficents):
  N  = int(np.sqrt(np.size(coefficents)))
  x  = cheb(N-1)[1]
  CP = np.polynomial.chebyshev.chebval(x, np.eye(N))
  CM = np.kron(CP, CP)  
  return np.reshape(CM.dot(coefficents),(N,N))

def extractcoeffs1D(function):
  N  = np.size(function) 
  x  = cheb(N-1)[1]
  CP = np.polynomial.chebyshev.chebval(x, np.eye(N))
  return np.linalg.solve(CP, function)

def extractvalues1D(coefficents):
  N  = np.size(coefficents)
  x  = cheb(N-1)[1]
  CP = np.polynomial.chebyshev.chebval(x, np.eye(N))
  print np.size(coefficents)
  print np.shape(CP)
  return CP.dot(coefficents)

#------------------------------------------------------------
# test utilities
#------------------------------------------------------------

if(0):
  x = cheb(10)[1]
  uu, vv = np.meshgrid(x,x)
  domain = np.exp(-uu**2.0/0.1)*np.exp(-vv**2.0/0.1)
  plt.figure(1)
  plt.imshow(domain)
  domainN = extractvalues(extractcoeffs(domain))
  plt.figure(2)
  plt.imshow(domainN)
  plt.show()
elif(0):
  x = cheb(10)[1]
  uu, vv = np.meshgrid(x,x)
  domain = np.exp(-uu**2.0/0.1)*np.exp(-vv**2.0/0.1)
  plt.figure(1)
  plt.imshow(domain)
  # add zero coefficents
  coefficentsN = np.pad(extractcoeffs(domain), (0, 23), 'constant')
  domainN = extractvalues(coefficentsN)
  plt.figure(2)
  plt.imshow(domainN)
  plt.show()
else:
  pass

if(0):
  x = cheb(10)[1]
  function = x**3.0 + 4*x**2.0 - x
  plt.plot(x, function)

  coefficents = extractcoeffs1D(function) # get the coefficents
  coefficentsN =  np.pad(coefficents, (0,20), 'constant')# pad the coefficents
  functionN   = extractvalues1D(coefficentsN) # extract the values
  plt.plot(cheb(np.size(coefficentsN)-1)[1], functionN, '-o')
  plt.show()