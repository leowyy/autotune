from ..core.CIFAR10_problem2 import *
import argparse
import pickle

# input_dir = '/Users/signapoop/Desktop/data/'
# output_dir = '/Users/signapoop/Desktop/autotune/autotune/sandpit/checkpoint/'

# input_dir = '/data/'
# output_dir = '/checkpoint/'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('-i', '--input_dir', type=str, help='input dir')
parser.add_argument('-o', '--output_dir', type=str, help='output dir')
args = parser.parse_args()

print(args.input_dir)
print(args.output_dir)

hps = ['learning_rate', 'n_units_1', 'n_units_2', 'n_units_3', 'batch_size']
problem = CIFAR10_problem2(args.input_dir, args.output_dir)
problem.print_domain()

x = problem.generate_random_arm(hps)
x['n_resources'] = 3
problem.eval_arm(x)

a = 42
filename = args.output_dir + 'test.pkl'
with open(filename, 'wb') as f:
    pickle.dump([a], f)


