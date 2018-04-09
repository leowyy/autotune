import time
import numpy as np
import GPyOpt
from GPyOpt.util.general import best_value
from GPyOpt.experiment_design import initial_design
from GPyOpt.core.task.space import Design_space

class RandomOptimiser():
    def __init__(self, f, domain, X_init = None, Y_init = None):
        self.f = f
        self.domain = Design_space(space = domain, constraints=None)
        self.X = X_init
        self.Y = Y_init

    def run_optimization(self, max_iter = None, max_time = np.inf, verbosity=False):
            
        # --- Setting up stop conditions
        if  (max_iter is None) and (max_time is None):
            self.max_iter = 0
            self.max_time = np.inf
        elif (max_iter is None) and (max_time is not None):
            self.max_iter = np.inf
            self.max_time = max_time
        elif (max_iter is not None) and (max_time is None):
            self.max_iter = max_iter
            self.max_time = np.inf
        else:
            self.max_iter = max_iter
            self.max_time = max_time

        # --- Initialize iterations and running time
        self.time_zero = time.time()
        self.cum_time  = 0
        self.num_iterations = 0
        self.checkpoints = []
        global_best = np.inf

        while (self.max_time > self.cum_time):
            if self.num_iterations > self.max_iter:
                print("Exceeded maximum number of iterations")
                break

            # Draw random sample
            X_new = self.draw_random_sample()
            Y_new = np.asscalar(self.f(X_new))

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
            self.num_iterations += 1
            self.checkpoints.append(self.cum_time)

            if verbosity:
                print("num iteration: {}, time elapsed: {:.2f}s, f_best: {:.5f}".format(
                    self.num_iterations, self.cum_time, global_best))

        self._compute_results()

    def draw_random_sample(self, n_samples = 1):
        return initial_design('random', self.domain, n_samples)

    def _compute_results(self):
        """
        Computes the optimum and its value.
        """
        self.Y_best = best_value(self.Y)
        self.x_opt = self.X[np.argmin(self.Y),:]
        self.fx_opt = min(self.Y)



