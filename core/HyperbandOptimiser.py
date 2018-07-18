import time
from math import log, ceil
import numpy as np
from RandomOptimiser import RandomOptimiser


class HyperbandOptimiser(RandomOptimiser):
    def __init__(self):
        super(HyperbandOptimiser, self).__init__()
        self.name = "Hyperband"

    def run_optimization(self, problem, n_resources=None, max_iter=None, eta=3, verbosity=False):
        # problem provides generate_random_arm and eval_arm(x)

        print("\n---- Running hyperband optimisation ----")
        print("Max iterations = {}".format(max_iter))
        print("Halving rate eta = {}".format(eta))
        print("----------------------------------------")

        # --- Initialize iterations and running time
        self.time_zero = time.time()
        self.cum_time = 0
        self.num_iterations = 0
        self.checkpoints = []

        logeta = lambda x: log(x)/log(eta)
        s_max = int(logeta(max_iter))  # number of unique executions of Successive Halving (minus one)
        if s_max >= 2:
            s_min = 2  # skip the rest of the brackets after s_min
        else:
            s_min = 0
        B = (s_max+1)*max_iter  # total number of iterations (without reuse) per execution of Succesive Halving (n,r)

        # Repeat outer loop twice
        for _ in range(2):
            # Begin Finite Horizon Hyperband outlerloop.
            for s in reversed(range(s_min, s_max+1)):
                n = int(ceil(int(B/max_iter/(s+1))*eta**s))  # initial number of configurations
                r = max_iter*eta**(-s)  # initial number of iterations to run configurations for

                # Begin Finite Horizon Successive Halving with (n,r)
                # arms = [ problem.generate_random_arm(problem.hps) for i in range(n) ]
                arms = problem.generate_arms(n, problem.hps)
                for i in range(s+1):
                    # Run each of the n_i configs for r_i iterations and keep best n_i/eta
                    n_i = n*eta**(-i)
                    r_i = r*eta**(i)
                    val_losses = []
                    test_losses = []

                    for arm in arms:
                        # Assign r_i units of resource to arm
                        # arm['n_resources'] = r_i
                        val_loss, test_loss = problem.eval_arm(arm, r_i)
                        val_losses.append(val_loss)
                        test_losses.append(test_loss)

                    # Track stats
                    min_val = min(val_losses)
                    min_index = val_losses.index(min_val)
                    best_arm = arms[min_index]
                    Y_new = test_losses[min_index]

                    # Update history
                    self.arms.append(best_arm)
                    self.val_loss.append(min_val)
                    self.Y.append(Y_new)

                    # --- Update current evaluation time and function evaluations
                    self.cum_time = time.time() - self.time_zero
                    self.checkpoints.append(self.cum_time)

                    if verbosity:
                        print("time elapsed: {:.2f}s, f_current: {:.5f}, f_best: {:.5f}".format(
                            self.cum_time, Y_new, min(self.Y)))

                    arms = [ arms[i] for i in np.argsort(val_losses)[0:int( n_i/eta )] ]
                #### End Finite Horizon Successive Halving with (n,r)

        self._compute_results()
