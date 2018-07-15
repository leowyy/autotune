import pickle
import argparse

from ..core.RandomOptimiser import RandomOptimiser
from ..core.SigOptimiser import SigOptimiser
from ..benchmarks.mnist_problem import MnistProblem
# from ..benchmarks.cifar_problem import CifarProblem
# from ..benchmarks.svhn_problem import SvhnProblem

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-i', '--input_dir', type=str, help='input dir')
parser.add_argument('-o', '--output_dir', type=str, help='output dir')
parser.add_argument('-res', '--n_resources', default=3, type=int, help='n_resources')
args = parser.parse_args()

print("Input directory: {}".format(args.input_dir))
print("Output directory: {}".format(args.output_dir))
print("# resources: {}".format(args.n_resources))

# Define problem instance
problem = MnistProblem(args.input_dir, args.output_dir)
problem.print_domain()

# Define maximum units of resource assigned to each optimisation iteration
n_resources = args.n_resources

random_opt = RandomOptimiser()
random_opt.run_optimization(problem, n_resources, max_iter=2, verbosity=True)

sig_opt = SigOptimiser()
sig_opt.run_optimization(problem, n_resources, max_iter=5, verbosity=True)

filename = args.output_dir + 'results.pkl'
with open(filename, 'wb') as f:
    pickle.dump([random_opt, sig_opt], f)

print(sig_opt.arm_opt)
print(sig_opt.fx_opt)
