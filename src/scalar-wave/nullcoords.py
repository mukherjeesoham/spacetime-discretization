#==========================================================
# Code to solve the scalar wave equation using the
# discretized action in null 1+1 Minkowski spacetime.
# Soham 9 2017
#==========================================================

import numpy as np
import utilities as util
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

	BC = "Sine"
	if BC == "Gaussian":
		b[:,  0] = np.exp(-10.0*u**2.0)	# -m
		b[0,  :] = np.exp(-10.0*v**2.0)	# +m
	if BC == "Sine":
		b[:,  0] = np.sin(np.pi*u)	# -m
		b[0,  :] = np.sin(np.pi*v)	# +m

	b = np.ravel(b)

	#------------------------------------------------
	# solve the system
	#------------------------------------------------
	
	print("Solving a dimension %r linear system..."%np.shape(A)[0])
	z = np.linalg.solve(A, b)
	zz = np.reshape(z, (N+1, N+1))

	#------------------------------------------------
	# Analysis
	#------------------------------------------------

	print "\nIs A symmetric?", np.allclose(A, A.T)

	def is_pos_def(x):
	    return np.all(np.linalg.eigvals(x) > 0)
	if(0):
		print "Is A positive definite?", is_pos_def(A)

	ss = 2*np.sin(np.pi*((vv + uu)/2.0))*np.cos(np.pi*((vv - uu)/2.0))

	ee = ss - zz
	L2 =  np.sqrt(np.trace(np.abs(W*np.ravel(ee**2.0))))
	print "L2 error norm: ", L2

	end = time.time()
	runtime = end - start
	return L2, zz, uu, vv, ss, runtime


#===================================================
# Call function
#===================================================

# single solutionba
if(0):
	L2, zz, uu, vv, ss, runtime =main(40)

	if(1):
		fig = plt.figure(2)
		ax = fig.gca(projection='3d')
		plt.xlabel(r"$u$")
		plt.ylabel(r"$v$")

		Z = cm.viridis_r((zz-zz.min())/(zz.max()-zz.min()))
		S = cm.viridis_r((ss-ss.min())/(ss.max()-ss.min()))

		if(0):
			surf = ax.plot_surface(uu, vv, ss, rstride=1, cstride=1,
	               facecolors = S, shade=False, linewidth=0.8)
		else:
			surf = ax.plot_surface(uu, vv, zz, rstride=1, cstride=1,
	               facecolors = Z, shade=False, linewidth=0.6)
		surf.set_facecolor((0,0,0,0))
		plt.show()

# convergence
if(1):
	N   = [3, 7, 15, 31, 63]
	Err  = np.zeros(len(N))
	Time = np.zeros(len(N))

	for _i, _n in enumerate(N):
		L2, uu, xx, tt, ss, runtime = main(_n)
		Err[_i]  = L2
		Time[_i] = runtime

	if(1):
		plt.semilogy(N, Err, 'm-o')
		plt.xlabel("N")
		plt.ylabel(r"$L_2$" + " norm")
	else:
		plt.plot(N, Time, 'm-o')
		plt.xlabel("N")
		plt.ylabel("Run time (s)")
	plt.show()










