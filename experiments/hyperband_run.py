import pickle
import argparse

from ..core.HyperbandOptimiser import HyperbandOptimiser
from ..core.RandomOptimiser import RandomOptimiser
# from ..benchmarks.mnist_problem import MnistProblem
from ..benchmarks.cifar_problem import CifarProblem

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-i', '--input_dir', type=str, help='input dir')
parser.add_argument('-o', '--output_dir', type=str, help='output dir')
parser.add_argument('-res', '--n_resources', default=3, type=int, help='n_resources')
args = parser.parse_args()

print("Input directory: {}".format(args.input_dir))
print("Output directory: {}".format(args.output_dir))
print("# resources: {}".format(args.n_resources))

# Define problem instance
problem = CifarProblem(args.input_dir, args.output_dir)
problem.print_domain()

# Define maximum units of resource assigned to each optimisation iteration
n_resources = args.n_resources

# # Run hyperband
hyperband_opt = HyperbandOptimiser()
hyperband_opt.run_optimization(problem, max_iter=n_resources, verbosity=True)

# Constrain random optimisation to the same time budget
time_budget = hyperband_opt.checkpoints[-1]
print("Time budget = {}s".format(time_budget))

filename = args.output_dir + 'results.pkl'
with open(filename, 'wb') as f:
    pickle.dump([hyperband_opt], f)