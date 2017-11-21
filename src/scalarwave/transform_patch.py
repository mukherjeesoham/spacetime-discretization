#=====================p==========================================
# Testing patch transformation properties
# Soham M 10/2017
#===============================================================

import numpy as np
from scalarwave_classes import patch, spec
import matplotlib.pyplot as plt

def transform(X, scale, shift):
	return (X + shift)/scale

def shrinkpatches(M, N):
	PATCH  = patch(N)
	X 	   = patch.chebnodes(PATCH)
	XX, YY = np.meshgrid(X, X)

	L = []
	for i in range(-M+1, M+1, 2):
		for j in range(-M+1, M+1, 2):
			L.append([i,-j])
	L = np.array(L)
	L = L[np.argsort(L[:, 1][::-1])]

	m = 0
	for index, val in np.ndenumerate(np.zeros((M, M))):
		print transform(X, M, L[m,0])
		print transform(X, M, L[m,1])
		XXS, YYS = np.meshgrid(transform(X, M, L[m,0]), transform(X, M, L[m,1]))
		plt.plot(XXS, YYS, 'm o')
		plt.plot(XX,  YY, 'r o')
		m += 1

		for i in np.linspace(-1,1, M+1)[1:-1]:
			plt.axvline([i], color='k', linestyle='-', linewidth=0.4)
			plt.axhline([i], color='k', linestyle='-', linewidth=0.4)
		plt.xlim(-1.25,1.25)
		plt.ylim(-1.25,1.25)
		plt.axhline([-1], linestyle='--')
		plt.axhline([1], linestyle='--')
		plt.axvline([-1], linestyle='--')
		plt.axvline([1], linestyle='--')
		plt.show()
	return None

shrinkpatches(2, 4)
