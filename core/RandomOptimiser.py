import time
import numpy as np
from ..util.best_value import best_value


class RandomOptimiser(object):
    def __init__(self, arms_init=[], val_loss_init=[], Y_init=[]):
        self.arms = arms_init
        self.val_loss = val_loss_init
        self.Y = Y_init

    def run_optimization(self, problem, n_resources, max_iter=None, max_time=np.inf, verbosity=False):
        # problem provides generate_random_arm and eval_arm(x)

        print("---- Running random optimisation ----")

        # --- Setting up stop conditions
        if (max_iter is None) and (max_time is None):
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

        while (self.max_time > self.cum_time):
            if self.num_iterations >= self.max_iter:
                print("Exceeded maximum number of iterations")
                break

            # Draw random sample
            arm = problem.generate_random_arm()
            arm['n_resources'] = n_resources  # Fix this

            # Evaluate arm on problem
            val_loss, Y_new = problem.eval_arm(arm)

            # Update history
            self.arms.append(arm)
            self.val_loss.append(val_loss)
            self.Y.append(Y_new)

            # --- Update current evaluation time and function evaluations
            self.cum_time = time.time() - self.time_zero
            self.num_iterations += 1
            self.checkpoints.append(self.cum_time)

            if verbosity:
                print("num iteration: {}, time elapsed: {:.2f}s, f_current: {:.5f}, f_best: {:.5f}".format(
                    self.num_iterations, self.cum_time, Y_new, min(self.Y)))

        self._compute_results()

    def _compute_results(self):
        """
        Computes the optimum and its value.
        """
        self.Y_best = best_value(self.Y)
        self.fx_opt = min(self.Y)
        self.arm_opt = self.arms[ self.Y.index(self.fx_opt) ]




