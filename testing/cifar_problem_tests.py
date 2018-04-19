from ..benchmarks.cifar_problem_1 import CifarProblem1
from ..benchmarks.cifar_problem_2 import CifarProblem2

data_dir = '/Users/signapoop/Desktop/data/'
output_dir = '/Users/signapoop/Desktop/autotune/autotune/experiments/checkpoint/'


def sampling_arms(n_arms):
    problem = CifarProblem1(data_dir, output_dir)
    problem.print_domain()

    for i in range(n_arms):
        arm = problem.generate_random_arm()
        print(arm)


if __name__ == "__main__":
    sampling_arms(5)
