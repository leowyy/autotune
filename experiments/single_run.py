import pickle
import argparse

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

arms = problem.generate_arms(1, problem.hps)
arm = arms[0]
# arm = problem.generate_random_arm(problem.hps)
# arm['n_resources'] = n_resources  # Fix this?

arm["batch_size"] = 32
arm["gamma"] = 0.1
arm["learning_rate"] = 0.0005171139689348882
arm["lr_step"] = 1
arm["momentum"] = 0.9
arm["n_units_1"] = 256.0
arm["n_units_2"] = 256.0
arm["n_units_3"] = 131.0
arm["weight_decay"] = 0.004

# Evaluate arm on problem
# val_loss, Y_new = problem.eval_arm(arm)
val_loss, Y_new = problem.eval_arm(arm, n_resources)

filename = args.output_dir + 'results.pkl'
with open(filename, 'wb') as f:
    pickle.dump([arm], f)



