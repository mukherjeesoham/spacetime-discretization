#==========================================================
# Code to solve the scalar wave equation using the
# discretized action by tiling up 1+1 Minkowski spacetime,
# using null coordinates, and multiple patches.
# Soham 9 2017
#==========================================================

import numpy as np
import sw_utilities as util
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def operator(N):
	Du, u  = util.cheb(N-1)
	Dv, v  = util.cheb(N-1)
	uu, vv = np.meshgrid(u,v)

	I  = np.eye(N)
	DU = np.kron(Du, I)
	DV = np.kron(I, Dv)

	D  = np.dot(DU,DV) + np.dot(DV,DU)
	V = np.outer(util.clencurt(N-1), util.clencurt(N-1))
	W = np.diag(np.ravel(V))
	A = W.dot(D)

	return A

def initialdata(u, v):
	return np.sin(np.pi*u)

def boundary(A, X, index, grid):

	N = int(np.power(np.size(A), 0.25))
	Du, u  = util.cheb(N-1)
	Dv, v  = util.cheb(N-1)

	BC = np.zeros((N ,N))
	BC[0, :] = BC[:, 0]  = 1

	A[np.where(np.ravel(BC)==1)[0]] = np.eye(N**2)[np.where(np.ravel(BC)==1)[0]]

	b  = np.zeros((N, N))

	if np.sum(index) == 0:				# it's the first box, relax.
		b[:,  0] = initialdata(u, v)
		b[0,  :] = initialdata(v, u)
	elif np.prod(index) == 0:			#it's at the boundary.
		if index[0] > index[1]:
			_i = index - [1,0]
			if grid[1][_i[0]][_i[1]] == 1:
				b[:, 0] = initialdata(u, v)
				b[0, :] = X[_i[0], _i[1]][-1, :]
		else:					
			_k = index - [0,1]
			if grid[1][_k[0]][_k[1]] == 1:
				b[0, :] = initialdata(v, u)
				b[:, 0] = X[_k[0], _k[1]][:, -1]
	else:								#it's somewhere in the middle. 
		_i = index - [1,0]
		_k = index - [0,1]
		if (grid[1][_i[0]][_i[1]] == 1) & (grid[1][_k[0]][_k[1]] == 1):
			b[0, :] = X[_i[0], _i[1]][-1, :]
			b[:, 0] = X[_k[0], _k[1]][:, -1]

	return A, np.ndarray.flatten(b)

def solve(N, index, X, grid):
	A    = operator(N)
	A, b = boundary(A, X, index, grid)
	x    = np.linalg.solve(A, b)
	X[index, :, :] = np.reshape(x, (N, N))
	return 1

def makeglobalgrid(M, N):
	grid = np.zeros((2, M, M))
	X    = np.zeros((M, M, N, N))

	for index, val in np.ndenumerate(grid[0]):
		grid[0][index] = np.sum(index)
	return grid, X

def assemblegrid(M,N,X):
	I = []
	for i in range(M):
		J = []
		for j in range(M):	
			J.append(X[i,j])
		I.append(J)
	return np.block(I)

def main(M,N):
	grid, X = makeglobalgrid(M,N)
	for i in range(int(np.max(grid[0]))+1):
		slice = np.transpose(np.where(grid[0]==i))
		for index in slice:
			grid[1][index[0]][index[1]] = solve(N, index, X, grid)
	return assemblegrid(M,N,X)

X = main(4,20)

if(1):
	plt.imshow(X)
	plt.show()

if(0):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	plt.xlabel(r"$u$")
	plt.ylabel(r"$v$")

	Z = cm.viridis_r((X-X.min())/(X.max()-X.min()))

	Du, u  = util.cheb(19)
	Dv, v  = util.cheb(19)
	uu, vv = np.meshgrid(u,v)

	surf = ax.plot_surface(uu, vv, X, rstride=1, cstride=1, 
		facecolors = Z, shade=False, linewidth=0.6)

	surf.set_facecolor((0,0,0,0))
	plt.show()

