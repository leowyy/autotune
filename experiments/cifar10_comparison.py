import pickle
import argparse
from ..core.CIFAR10_problem import *
from ..core.HyperbandOptimiser import *
from ..core.RandomOptimiser import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('-i', '--input_dir', type=str, help='input dir')
parser.add_argument('-o', '--output_dir', type=str, help='output dir')
args = parser.parse_args()

#input_dir = '/Users/signapoop/Desktop/autotune/data/'
#output_dir = '/Users/signapoop/Desktop/autotune/sandpit/checkpoint/'

# Define problem instance
problem = CIFAR10_problem(args.input_dir, args.output_dir)
problem.print_domain()

n_units = 81 # units of resource assigned to each random optimisation iteration

hyperband_opt = HyperbandOptimiser(problem)
hyperband_opt.run_optimization(max_iter = 81, verbosity = True)

time_budget = hyperband_opt.checkpoints[-1]
print("Time budget = {}".format(time_budget))

random_opt = RandomOptimiser(problem)
random_opt.run_optimization(n_units, max_time = time_budget, verbosity = True)

filename = args.output_dir + 'cifar.pkl'
with open(filename, 'wb') as f:
    pickle.dump([hyperband_opt.arms, hyperband_opt.checkpoints, hyperband_opt.Y, random_opt.arms, random_opt.checkpoints, random_opt.Y], f)