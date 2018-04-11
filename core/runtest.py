import pickle
import os
import matplotlib.pyplot as plt
from utils import best_value
# from CIFAR10_problem import *
# import argparse

# parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('-i', '--input_dir', type=str, help='input dir')
# parser.add_argument('-o', '--output_dir', type=str, help='output dir')
# args = parser.parse_args()
#
# def main():
#     a = [5,6,3,4,7,2,1,3,23,1,0,2]
#     print(best_value(a))
#
# main()
#
# print(args.input_dir)
# print(args.output_dir)
# filename = args.output_dir + 'myfile.txt'
# with open(filename, 'a') as f:
#     f.write('Please save me!\n')
path = '/Users/signapoop/Desktop/autotune/autotune/data'
os.chdir(path)
file = open("cifar.pkl",'rb')
object_file = pickle.load(file)
file.close()

# Initialise plot
hyperband = {}
random = {}
hyperband['checkpoints'] = object_file[1]
hyperband['Y'] = object_file[2]
random['checkpoints'] = object_file[4]
random['Y'] = object_file[5]

fig, ax = plt.subplots(1,1, figsize=(6, 4), dpi=100)
ax.plot(hyperband['checkpoints'], best_value(hyperband['Y']), '--bs', label='hyperband_opt')
ax.plot(random['checkpoints'], best_value(random['Y']), '--kx', label='random_opt')
plt.ylabel('Min Validation Error'); plt.xlabel('Time (s)');
plt.legend()
plt.show()