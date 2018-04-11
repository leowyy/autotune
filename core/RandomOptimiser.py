import time
import numpy as np
from utils import best_value

class RandomOptimiser(object):
    def __init__(self, problem, arms_init = [], Y_init = []):
        self.problem = problem      # problem provides generate_random_arm and eval_arm(x)
        self.arms = arms_init
        self.Y = Y_init

    def run_optimization(self, n_units, max_iter = None, max_time = np.inf, verbosity=False):

        print("---- Running random optimisation ----")

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
            arm = self.problem.generate_random_arm()

            arm['n_units'] = n_units # Fix this

            # Evaluate arm on problem
            Y_new = self.problem.eval_arm(arm)

            # Update history
            self.arms.append(arm)
            self.Y.append(Y_new)

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

    def _compute_results(self):
        """
        Computes the optimum and its value.
        """
        self.Y_best = best_value(self.Y)
        self.fx_opt = min(self.Y)
        self.arm_opt = self.arms[ self.Y.index(self.fx_opt) ]




