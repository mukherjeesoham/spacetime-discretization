"""
Solve a scalar wave equation [test] using 
spacetime discretization and FEM.
Domain: [-1, 1] x (0, 1]
Soham 6 2017

XXX: This is a very rudimentary version of the code to test 
the idea. A decent FEM code should allow for 
	> Ability to solve in higher dimenions (2, 3).
	> computing M and A from general basis functions (for eg. Chebyshef)
		> Requires numeric integration routines.
	> Enforing Dirichlet boundary conditions explicitly.
Additional features to be added to the list.
"""

import numpy as np
import matplotlib.pyplot as plt

def mesh(N, M):
	"""
	returns the nodes in spacetime.
	"""
	return np.linspace(-1, 1, N), np.linspace(0, 1, M)

def compute_MA(N):
	"""
	Construct the mass (M) and the stiffness (A) matrices
	given the number of points in the spatial dimension (N)
	"""
	M  = np.zeros((N-2, N-2))	# mass matrix [interior points]
	A =  np.zeros((N-2, N-2))	# stifness matrix [interior points]
	dx = 2.0/N 					# uniform in [-1, 1]

	for index, val in np.ndenumerate(M):
		i, j = index[0], index[1]
		if i==j:
			M[index] = (2.0*dx)/3
			A[index] = 2.0/dx
		elif (i + 1 == j) or (i - 1 == j):
			M[index] = dx/6.0
			A[index] = -1.0/dx
	return M, A

def hat(l, x, N):
	"""
	Takes the node number (l), input variable (x) and number of nodes
	Returns the value of the hat function phi(x) or psi(t)
	given the node at which it peaks, and value of x.
	"""
	nl = np.linspace(-1, 1, N)
	if (nl[l-1] <= x <= nl[l]):
		return (x - nl[l-1])/(nl[l] - nl[l-1])
	elif (nl[l] <= x <= nl[l+1]):
		return (nl[l+1] - x)/(nl[l+1] - nl[l])
	else:
		return 0

def get_coefficents(x, f):
	"""
	Given a function u(x,t) at an instant of time,
	find out c_{i} s.t. u{x_j} = \Sum_{i} c_{i} phi_{i}(x_j).
	"""
	N   = len(x)
	Phi = np.zeros((len(x)-2,len(x)-2))	# [interior points]

	for i in range(1, N-1):	
		for j in range(1, N-1):
			Phi[i-1,j-1] = hat(i, x[j], N)
	c = np.linalg.solve(Phi, f[1:-1])	# [interior points]

	#return after padding with zeros.
	return np.lib.pad(c, (1,1), 'constant', constant_values=(0))

def construct_function(x, c):
	"""
	Given c_{i} find u{x_j} = \Sum_{i} c_{i} phi_{i}(x_j).
	"""
	N   = len(x)
	Phi = np.zeros((len(x)-2,len(x)-2))	# [interior points]

	for i in range(1, N-1):
		for j in range(1, N-1):
			Phi[i-1,j-1] = hat(i, x[j], N)

	f = np.dot(Phi, c[1:-1])	# [interior points]

	#return after padding with zeros.
	return np.lib.pad(f, (1,1), 'constant', constant_values=(0))  

def assemble(x, f0, f1, M, M_ij, A_ij):
	u0   = get_coefficents(x, f0)[1:-1]	# get the interior points	
	u1   = get_coefficents(x, f1)[1:-1]	# get the interior points
	h0   = 1.0/M

	A = M_ij + ((h0**2.0)/6.0)*A_ij
	b = 2.0*np.dot(M_ij - ((2*h0**2.0)/3)*A_ij, u1) - np.dot(M_ij + ((h0**2.0)/6)*A_ij, u0)
	return A, b

def solve(A, b, x):
	x = x[1:-1]
	u2 = np.linalg.solve(A, b)
	f = construct_function(x, u2)
	return f

def main():
	N = 100	# nodes in interior. We are going to solve a 2x2 matrix.
	M = 10	# Doesn't seem to do much other than set h0.
	x, t = mesh(N, M)

	# give initial conditions;
	f0 = np.exp(np.exp((-x**2.0)/0.05))
	f1 = np.exp(np.exp((-x**2.0)/0.04))

	# compute the stiffness and mass matrix
	M_ij, A_ij = compute_MA(N)

	# find storage
	sol = np.zeros((M+2, N)) 

	# for all t > t[1], find the solution
	for tl in range(2, len(t)):
		print "In time-level: %r out of %r" %(tl, len(t))
		A, b = assemble(x, f0, f1, M, M_ij, A_ij)
		f2   = solve(A, b, x)	# f is of length N-2

		#enforce BCs by padding with 0 
		f2 = np.lib.pad(f2, (1,1), 'constant', constant_values=(0))	# now f is of length N
		sol[tl-2] = f2 #store solution

		#update vectors
		f0 = f1
		f1 = f2

	return x, t, sol
#--------------------------------------------------------
# Call solver
#--------------------------------------------------------

if(1):
	x, t, sol = main()

# the saddest way to check output
for slice in sol:
	plt.plot(x, slice)

plt.show()

#--------------------------------------------------------
# Test functions
#--------------------------------------------------------

N = 10
x = np.linspace(-1, 1, N)

if(1):	#test hat functions
	y = np.zeros(N)
	plt.plot(x, np.zeros(len(x)), 'o')
	for i in range(1, N-1):
		for w, _x in enumerate(x):
			y[w] = hat(i, _x, N)
		plt.plot(x,y)
	plt.show()

if(0):
	f = -x**2.0
	print get_coefficents(x, f)

if(0):
	# FIXME: Check why things aren't working for 
	# 		 arbitrary input vector lengths.
	f  = np.exp(-x**2.0/0.04)
	c  = get_coefficents(x, f)
	f_approx = construct_function(x, c)
	plt.plot(x, f_approx, 'r-')
	plt.show()

if(0):
	M = 4
	x, t = mesh(N, N)

	# give initial conditions;
	f0 = np.exp(np.exp((-x**2.0)/0.05)) 	# u_{r-1}	
	f1 = np.exp(np.exp((-x**2.0)/0.04))		# u_{r}	

	# compute the stiffness and mass matrix
	M_ij, A_ij = compute_MA(N)

	A, b = assemble(x, f0, f1, M, M_ij, A_ij)

	# test solve
	f = solve(A, b, x)