import numpy as np
import utilities as util
import matplotlib.pyplot as plt

def init(func):					# returns Boundary
	bnd = func(dictionary["chebnodes"])
	return bnd

def solve(bnd1, bnd2):          # returns Patch
	N = dictionary["size"]
	A = dictionary["operator"]
	b = util.makeboundaryvec(N, bnd1, bnd2)
	patch = np.reshape(np.linalg.solve(A, b), (N+1, N+1))
	return patch

def extract(patch, int):		# returns Boundary
	if int == 0:
		return patch[:,  0]
	else:
		return patch[0,  :]

def output(patch):              # returns None
    return None

dictionary = {					# make a global dictionary all functions can access
	"size"      : 40,
	"chebnodes" : util.cheb(40)[1],
	"operator"  : util.operator(40)
}

"""
To solve for the first patch, we follow
    init < solve < output
To solve for subsequent patches
    extract < solve < output
"""

# solve for a single patch
patch = solve(init(util.makeinitialdata), init(util.makeinitialdata))

plt.imshow(patch)
plt.show()
