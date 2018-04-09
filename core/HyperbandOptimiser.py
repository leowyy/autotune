import time
from math import log, ceil
import numpy as np
from RandomOptimiser import RandomOptimiser

class HyperbandOptimiser(RandomOptimiser):
	def __init__(self, f, domain, X_init = None, Y_init = None):
		super(HyperbandOptimiser, self).__init__(f, domain, X_init, Y_init)

	def run_optimization(self, max_iter = None, eta = 4, verbosity=False):

		# --- Initialize iterations and running time
		self.time_zero = time.time()
		self.cum_time  = 0
		self.num_iterations = 0
		self.checkpoints = []
		global_best = np.inf

		logeta = lambda x: log(x)/log(eta)
		s_max = int(logeta(max_iter))  # number of unique executions of Successive Halving (minus one)
		B = (s_max+1)*max_iter  # total number of iterations (without reuse) per execution of Succesive Halving (n,r)

		#### Begin Finite Horizon Hyperband outlerloop. Repeat indefinetely.
		for s in reversed(range(s_max+1)):
			n = int(ceil(int(B/max_iter/(s+1))*eta**s)) # initial number of configurations
			r = max_iter*eta**(-s) # initial number of iterations to run configurations for

			#### Begin Finite Horizon Successive Halving with (n,r)
			T = [ self.draw_random_sample() for i in range(n) ] 
			for i in range(s+1):
				# Run each of the n_i configs for r_i iterations and keep best n_i/eta
				n_i = n*eta**(-i)
				r_i = r*eta**(i)
				val_losses = []

				for config in T:
					# Overwrite n_iters by r_i !!!!
					config[:,2] = r_i
					loss = np.asscalar(self.f(config))
					val_losses.append(loss)
				
				# Track stats
				min_index = np.argmin(val_losses)
				X_new = T[min_index]
				Y_new = val_losses[min_index]

				if self.X == None:
					self.X = X_new
					self.Y = Y_new

				else:
					self.X = np.vstack((self.X, X_new))
					self.Y = np.vstack((self.Y, Y_new))

				if Y_new < global_best:
					global_best = Y_new

				# --- Update current evaluation time and function evaluations
				self.cum_time = time.time() - self.time_zero
				self.checkpoints.append(self.cum_time)

				if verbosity:
					print("time elapsed: {:.2f}s, f_best: {:.5f}".format(
						self.cum_time, global_best))
				
				T = [ T[i] for i in np.argsort(val_losses)[0:int( n_i/eta )] ]
			#### End Finite Horizon Successive Halving with (n,r)

		self._compute_results()
