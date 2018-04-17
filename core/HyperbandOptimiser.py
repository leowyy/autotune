import time
from math import log, ceil
import numpy as np
from RandomOptimiser import RandomOptimiser

class HyperbandOptimiser(RandomOptimiser):
    def __init__(self, arms_init=[], Y_init=[]):
        super(HyperbandOptimiser, self).__init__(arms_init, Y_init)
        # problem provides generate_random_arm and eval_arm(x)

    def run_optimization(self, problem, n_units=None, max_iter=None, eta=3, verbosity=False):

        print("---- Running hyperband optimisation ----")

        # --- Initialize iterations and running time
        self.time_zero = time.time()
        self.cum_time  = 0
        self.num_iterations = 0
        self.checkpoints = []
        global_best = np.inf

        logeta = lambda x: log(x)/log(eta)
        s_max = int(logeta(max_iter))  # number of unique executions of Successive Halving (minus one)
        s_min = 2  # skip the rest of the brackets after s_min
        B = (s_max+1)*max_iter  # total number of iterations (without reuse) per execution of Succesive Halving (n,r)

        #### Begin Finite Horizon Hyperband outlerloop. Repeat indefinetely.
        for s in reversed(range(s_min, s_max+1)):
            n = int(ceil(int(B/max_iter/(s+1))*eta**s))  # initial number of configurations
            r = max_iter*eta**(-s)  # initial number of iterations to run configurations for

            #### Begin Finite Horizon Successive Halving with (n,r)
            arms = [ problem.generate_random_arm() for i in range(n) ]
            for i in range(s+1):
                # Run each of the n_i configs for r_i iterations and keep best n_i/eta
                n_i = n*eta**(-i)
                r_i = r*eta**(i)
                val_losses = []

                for arm in arms:
                    # Assign r_i units of resource to arm
                    arm['n_resources'] = r_i
                    loss = problem.eval_arm(arm)
                    val_losses.append(loss)

                # Track stats
                Y_new = min(val_losses)
                min_index = val_losses.index(Y_new)
                best_arm = arms[min_index]

                # Update history
                self.arms.append(best_arm)
                self.Y.append(Y_new)

                if Y_new < global_best:
                    global_best = Y_new

                # --- Update current evaluation time and function evaluations
                self.cum_time = time.time() - self.time_zero
                self.checkpoints.append(self.cum_time)

                if verbosity:
                    print("time elapsed: {:.2f}s, f_best: {:.5f}".format(
                        self.cum_time, global_best))

                arms = [ arms[i] for i in np.argsort(val_losses)[0:int( n_i/eta )] ]
            #### End Finite Horizon Successive Halving with (n,r)

        self._compute_results()
