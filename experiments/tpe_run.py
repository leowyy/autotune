import pickle
import argparse

from ..core.HyperbandOptimiser import HyperbandOptimiser
from ..core.TpeOptimiser import TpeOptimiser
from ..core.RandomOptimiser import RandomOptimiser
# from ..benchmarks.mnist_problem import MnistProblem
# from ..benchmarks.cifar_problem import CifarProblem
# from ..benchmarks.svhn_problem import SvhnProblem
from ..benchmarks.mrbi_problem import MrbiProblem


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-i', '--input_dir', type=str, help='input dir')
parser.add_argument('-o', '--output_dir', type=str, help='output dir')
parser.add_argument('-res', '--n_resources', default=3, type=int, help='n_resources')
args = parser.parse_args()

print("Input directory: {}".format(args.input_dir))
print("Output directory: {}".format(args.output_dir))
print("# resources: {}".format(args.n_resources))

# Define problem instance
problem = MrbiProblem(args.input_dir, args.output_dir)
problem.print_domain()
print(problem.hps)

# Define maximum units of resource assigned to each optimisation iteration
n_resources = args.n_resources

# Run hyperband
# hyperband_opt = HyperbandOptimiser()
# hyperband_opt.run_optimization(problem, max_iter=n_resources, verbosity=True)

# Run random
random_opt = RandomOptimiser()
#time = 3600 * 3
#random_opt.run_optimization(problem, n_resources, max_time=time, verbosity=True)
random_opt.run_optimization(problem, n_resources, max_iter=48, verbosity=True)

# Run tpe
# tpe_opt = TpeOptimiser()
# tpe_opt.run_optimization(problem, n_resources, max_time=7500, verbosity=True)

filename = args.output_dir + 'results.pkl'
with open(filename, 'wb') as f:
    # pickle.dump([tpe_opt, tpe_opt2, tpe_opt3, tpe_opt4, tpe_opt5], f)
    pickle.dump([random_opt], f)
    # pickle.dump([hyperband_opt, random_opt, tpe_opt], f)