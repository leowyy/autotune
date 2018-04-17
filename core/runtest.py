from CIFAR10_problem2 import *
input_dir = '/Users/signapoop/Desktop/data/'
output_dir = '/Users/signapoop/Desktop/autotune/autotune/sandpit/checkpoint/'

hps = ['learning_rate', 'n_units_1', 'n_units_2', 'n_units_3', 'batch_size']
problem = CIFAR10_problem2(input_dir, output_dir)
problem.print_domain()

x = problem.generate_random_arm(hps)
x['n_resources'] = 81
problem.eval_arm(x)


