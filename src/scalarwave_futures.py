#--------------------------------------------------------------------
# Spacetime Discretization methods Scalar Wave Prototype 
# Utilities for handling futures
# Soham M 10-2017
# UNDER CONSTRUCTION
#--------------------------------------------------------------------

import numpy as np
import concurrent.futures
from scalarwave_patch import patch

class future(object):

	@static_method
	def future_set_boundary(func):                	
	    return executor.submit(init, func)

	@static_method
	def future_patch(operator, boundary_conditions):       	
	    return executor.submit(patch.solve, operator.result(), boundary_conditions.result())

	@static_method
	def future_extract_boundary(fpatch, col):          
		return executor.submit(patch.extractpatchBC, fpatch.result(), 1)
