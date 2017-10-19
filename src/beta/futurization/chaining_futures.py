from concurrent.futures import ThreadPoolExecutor
import numpy as np

P = np.zeros(3)

iter = 0
# to check if futures are acutally chained or overwritten.
with ThreadPoolExecutor(max_workers=4) as executor:
	future = executor.submit(pow, 2, 2)
	print iter, future.result()
	for iter in range(1,3):
		print 20*"-"
		print iter, future.result()
		P[iter] = executor.submit(pow, future.result(), iter)
		future = executor.submit(pow, P[iter].result(), iter + 1)
		print iter, future.result()
		# future = future.result() + future.result()
  
	print "future.result() : ", future.result()