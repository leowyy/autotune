from __future__ import division
import numpy as np
from ..core.problem_def import Problem
from ..core.params import *


class BraninProblem(Problem):

    def __init__(self):
        self.domain = self.initialise_domain()
        self.hps = None
        self.name = "Branin"

    def eval_arm(self, arm, n_resources):
        self.a = 1
        self.b = 5.1/(4*np.pi**2)
        self.c = 5/np.pi
        self.r = 6
        self.s = 10
        self.t = 1/(8*np.pi)

        x1 = arm["x"]
        x2 = arm["y"]
        fval = self.a * (x2 - self.b*x1**2 + self.c*x1 - self.r)**2 + self.s*(1-self.t)*np.cos(x1) + self.s
        return fval, fval

    def initialise_domain(self):
        params = {
            'x': Param('x', -5, 10, distrib='uniform', scale='linear'),
            'y': Param('y', 1, 15, distrib='uniform', scale='linear'),
            }
        return params

    def construct_arms(self, arms):
        return arms


if __name__ == "__main__":
    problem = BraninProblem()
    # arm = {"x": np.pi, "y": 2.275}
    arms = problem.generate_arms(1)
    val, _ = problem.eval_arm(arms[0], 0)
    print(arms[0])
    print(val)
