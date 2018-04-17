import numpy as np
from ..core.CIFAR10_problem import *

# Define problem instance
dirname = '/Users/signapoop/Desktop/autotune/autotune/sandpit/checkpoint/'
data_dir = '/Users/signapoop/Desktop//data'
problem = CIFAR10_problem(data_dir, dirname)
problem.print_domain()

x = np.array([81., -3.41513064, 2.04309237, -3.03636196, 0.48027257])
x = np.reshape(x, (1, -1))
problem.f(x)
