import argparse

# from ..benchmarks.mnist_problem import MnistProblem
# from ..benchmarks.cifar_problem_1 import CifarProblem1
# from ..benchmarks.cifar_problem_2 import CifarProblem2
from ..benchmarks.cifar_problem_new import CifarProblemNew


class ProblemTest(object):
    def __init__(self, data_dir, output_dir):
        self.problem = CifarProblemNew(data_dir, output_dir)
        self.problem.print_domain()

    def sampling_arms(self, n_arms):
        arms = self.problem.generate_arms(n_arms, self.problem.hps)
        return arms

    def run_test(self, arms, n_resources):
        for arm in arms:
            print(arm)
            self.problem.eval_arm(arm, n_resources)


if __name__ == "__main__":
    data_dir = '/Users/signapoop/Desktop/data/'
    output_dir = '/Users/signapoop/Desktop/autotune/autotune/experiments/checkpoint/exp1/'

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('-i', '--input_dir', default=data_dir, type=str, help='input dir')
    parser.add_argument('-o', '--output_dir', default=output_dir, type=str, help='output dir')
    args = parser.parse_args()

    print(args.input_dir)
    print(args.output_dir)

    problem_test = ProblemTest(args.input_dir, args.output_dir)
    arms = problem_test.sampling_arms(1)
    problem_test.run_test(arms, 27)
    print("\nPausing and resuming........................\n")
    problem_test.run_test(arms, 1)


