# We'll try to construct a lambda function in a loop

import numpy as np

f = lambda p: 1
xp = lambda x, p: x**p

for p in range(3):
	f = f + xp(x, p)
