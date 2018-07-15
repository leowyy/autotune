import time
import numpy as np
from RandomOptimiser import RandomOptimiser
from sigopt import Connection


class SigOptimiser(RandomOptimiser):
    def __init__(self):
        super(SigOptimiser, self).__init__()
        self.name = "SigOpt"

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

        print("\n---- Running SigOpt optimisation ----")
        print("Resource per iteration = {}".format(n_resources))
        print("Max iterations = {}".format(max_iter))
        print("Max time  = {}s".format(max_time))
        print("----------------------------------------")

        # --- Initialize iterations and running time
        self.time_zero = time.time()
        self.cum_time = 0
        self.num_iterations = 0
        self.checkpoints = []

        # Wrap parameter space
        space = self.initialise_sigopt_space(problem)

        # Wrap function
        objective_fn = lambda arm: self.initialise_sigopt_objective(problem, n_resources, arm)

        # Create SigOpt experiment
        conn = Connection(client_token="RAGFJSAISOJGFQOXCAVIVQRNNGOQNYGDEYISHTETQZCNWJNA")
        experiment = conn.experiments().create(
            name=problem.name,
            parameters=space,
            observation_budget=max_iter
        )
        print("Created experiment: https://sigopt.com/experiment/" + experiment.id)

        # Clear any open suggestions
        conn.experiments(experiment.id).suggestions().delete(state="open")

        while (self.max_time > self.cum_time):
            if self.num_iterations >= self.max_iter:
                print("Exceeded maximum number of iterations")
                break

            # Draw sample using SigOpt suggestion service
            suggestion = conn.experiments(experiment.id).suggestions().create()
            arm = suggestion.assignments

            # Evaluate arm on problem
            val_loss, Y_new = objective_fn(arm)

            # Update history
            self.arms.append(arm)
            self.val_loss.append(val_loss)
            self.Y.append(Y_new)

            # Add observation to SigOpt history
            # NB: SigOpt solves a maximisation problem, so it is important to negate the val loss
            conn.experiments(experiment.id).observations().create(
                suggestion=suggestion.id,
                value=-1 * val_loss,
            )

            # --- Update current evaluation time and function evaluations
            self.cum_time = time.time() - self.time_zero
            self.num_iterations += 1
            self.checkpoints.append(self.cum_time)

            if verbosity:
                print("num iteration: {}, time elapsed: {:.2f}s, f_current: {:.5f}, f_best: {:.5f}".format(
                    self.num_iterations, self.cum_time, Y_new, min(self.Y)))

        self._compute_results()

    def initialise_sigopt_objective(self, problem, n_resources, params):
        # params is a dict: name -> value
        # instantiate default parameters if any are required
        if problem.hps is not None:
            for p in problem.domain.keys():
                if p not in problem.hps:
                    val = problem.domain[p].init_val
                    assert val is not None, "No default value is set for param {}".format(p)
                    params[p] = val

        def apply_logarithms(arm):
            for p_name in arm.keys():
                if p_name[:8] == "int_log_":
                    arm[p_name[8:]] = round(np.exp(arm[p_name]))
                elif p_name[:4] == "log_":
                    arm[p_name[4:]] = np.exp(arm[p_name])
            return arm

        # Apply transformations to log params
        params = apply_logarithms(params)
        # create model file
        arms = problem.construct_arms([params])
        # run model
        val_loss, Y_new = problem.eval_arm(arms[0], n_resources)
        return val_loss, Y_new

    def initialise_sigopt_space(self, problem):
        def sigopt_param_converter(hb_param):
            name = hb_param.name
            min_val = hb_param.get_min()
            max_val = hb_param.get_max()
            interval = hb_param.interval
            if hb_param.scale == "log":
                assert hb_param.logbase == np.e
                name = "log_" + name
                if interval:
                    name = "int_" + name
            param_type = 'double'
            if interval and hb_param.scale != "log":
                param_type = 'int'
            return dict(name=name, type=param_type, bounds=dict(min=min_val, max=max_val))
        space = []
        for p in problem.domain.keys():
            if problem.hps is None or p in problem.hps:
                space.append(sigopt_param_converter(problem.domain[p]))
        print(space)
        return space

