#==========================================================
# Code to solve the scalar wave equation using the
# discretized action in null 1+1 Minkowski spacetime, using
# null coordinates.
# Soham 9 2017
#==========================================================

import numpy as np
import sw_utilities as util
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import matplotlib 

# Creates N+1 points.
def main(N):
	start = time.time()
	#------------------------------------------------
	# Grid
	#------------------------------------------------
	Du, u  = util.cheb(N)
	Dv, v  = util.cheb(N)
	uu, vv = np.meshgrid(u,v)

	#------------------------------------------------
	# Construct derivate + integral operators
	#------------------------------------------------
	I  = np.eye(N+1)

	DU = np.kron(Du, I)
	DV = np.kron(I, Dv)

	# note that the '2' comes from the summation over \eta^{ab}.
	D  = np.dot(DU,DV) + np.dot(DV,DU)

	# construct the weight matrix
	V = np.outer(util.clencurt(N), util.clencurt(N))
	W = np.diag(np.ravel(V))

	# construct the main operator
	A = W.dot(D)
	
	#------------------------------------------------
	# Impose boundary conditions and construct b
	#------------------------------------------------

	BC = np.zeros((N+1,N+1))

	# Dirichlet boundary conditions
	BC[0, :] = BC[:, 0]  = 1	
	A[np.where(np.ravel(BC)==1)[0]] = np.eye((N+1)**2)[np.where(np.ravel(BC)==1)[0]]

	# set the Dirichlet boundary values
	b = np.zeros((N+1, N+1))

	BC = "forced-potential"
	if BC == "Gaussian":
		b[:,  0] = np.exp(-10.0*u**2.0)	# -m
		b[0,  :] = np.exp(-10.0*v**2.0)	# +m
	if BC == "Sine":
		b[:,  0] = np.sin(np.pi*u)	# -m
		b[0,  :] = np.sin(np.pi*v)	# +m
	if BC == 'forced-potential':
		for index, value in np.ndenumerate(b):
			t = (u[index[0]] + v[index[1]])/2.0
			r = (v[index[0]] - u[index[1]])/2.0
			b[index] = np.cos(4*np.pi*t)*np.exp(-t**2.0)*np.exp(-r**2.0/(0.01))
		plt.imshow(b)
		plt.show()
		# b[:,  0] = 0.0	# -m
		# b[0,  :] = 0.0	# +m

	b = np.ravel(b)

	#------------------------------------------------
	# solve the system
	#------------------------------------------------
	
	print("Solving a dimension %r linear system..."%np.shape(A)[0])
	z = np.linalg.solve(A, b)
	zz = np.reshape(z, (N+1, N+1))
	end = time.time()
	runtime = end - start
	return zz, uu, vv


#===================================================
# Call function
#===================================================

zz, uu, vv =main(60)

if(1):
	fig = plt.figure(1)
	plt.imshow(zz)
	plt.axis("off")
	plt.show()
else:
	fig = plt.figure(2)
	ax = fig.gca(projection='3d')
	plt.xlabel(r"$u$")
	plt.ylabel(r"$v$")

	Z = cm.viridis_r((zz-zz.min())/(zz.max()-zz.min()))
	surf = ax.plot_surface(uu, vv, zz, rstride=1, cstride=1,
		facecolors = Z, shade=False, linewidth=0.6)
	surf.set_facecolor((0,0,0,0))
	plt.show()




