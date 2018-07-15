import pickle
import argparse

from ..core.RandomOptimiser import RandomOptimiser
from ..core.SigOptimiser import SigOptimiser
from ..benchmarks.branin_problem import BraninProblem


# Define problem instance
problem = BraninProblem()
problem.print_domain()

# Define maximum units of resource assigned to each optimisation iteration
n_resources = 0

random_opt = RandomOptimiser()
random_opt.run_optimization(problem, n_resources, max_iter=30, verbosity=True)

sig_opt = SigOptimiser()
sig_opt.run_optimization(problem, n_resources, max_iter=30, verbosity=True)

print(random_opt.arm_opt)
print(random_opt.fx_opt)

print(sig_opt.arm_opt)
print(sig_opt.fx_opt)

# print(sig_opt.arms)
# print(sig_opt.Y)
# print(sig_opt.Y_best)


