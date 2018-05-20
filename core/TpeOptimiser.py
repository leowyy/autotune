import time
from hyperopt_source import fmin, tpe, hp, Trials, STATUS_OK
import numpy as np
from functools import partial
from RandomOptimiser import RandomOptimiser


class TpeOptimiser(RandomOptimiser):
    def __init__(self, arms_init=[], val_loss_init=[], Y_init=[]):
        super(TpeOptimiser, self).__init__(arms_init, val_loss_init, Y_init)
        self.name = "TPE"

    def run_optimization(self, problem, n_resources, max_iter=None, max_time=np.inf, verbosity=False):
        # problem provides generate_random_arm and eval_arm(x)
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

        print("\n---- Running TPE optimisation ----")
        print("Resource per iteration = {}".format(n_resources))
        print("Max iterations = {}".format(max_iter))
        print("Max time  = {}s".format(max_time))
        print("----------------------------------------")

        # --- Initialize iterations and running time
        self.time_zero = time.time()
        self.cum_time = 0
        self.num_iterations = 0
        trials = Trials()

        # Wrap parameter space
        space = self.initialise_hyperopt_space(problem)

        # Wrap function
        objective_fn = lambda arm: self.initialise_hyperopt_objective(problem, n_resources, arm)

        # Run optimiser
        best = fmin(objective_fn,
                    space,
                    algo=partial(tpe.suggest, n_startup_jobs=10),
                    max_evals=self.max_iter,
                    max_time=self.max_time,
                    trials=trials,
                    verbose=verbosity)

        # Compute statistics
        self.arms = []
        self.Y = []
        self.val_loss = []
        self.checkpoints = []
        for t in trials.trials:
            self.arms.append(t['misc']['vals'])
            self.Y.append(t['result']['test_loss'])
            self.val_loss.append(t['result']['loss'])
            self.checkpoints.append(t['result']['eval_time'] - self.time_zero)

        self.trials = trials
        self._compute_results()

    def initialise_hyperopt_objective(self, problem, n_resources, params):
        # create model file
        arms = problem.construct_arms([params])
        # run model
        val_loss, Y_new = problem.eval_arm(arms[0], n_resources)
        return {'loss': val_loss, 'status': STATUS_OK, 'test_loss': Y_new, 'eval_time': time.time()}

    def initialise_hyperopt_space(self, problem):
        def hyperopt_param_converter(hb_param):
            name = hb_param.name
            min_val = hb_param.get_min()
            max_val = hb_param.get_max()
            interval = hb_param.interval
            if hb_param.scale == "log":
                assert hb_param.logbase == np.e
                if interval:
                    return hp.qloguniform(name, min_val, max_val, interval)
                else:
                    return hp.loguniform(name, min_val, max_val)
            else:
                if interval:
                    return hp.quniform(name, min_val, max_val, interval)
                else:
                    return hp.uniform(name, min_val, max_val)

        space = {}
        for p in problem.domain.keys():
            space[p] = hyperopt_param_converter(problem.domain[p])
        return space
