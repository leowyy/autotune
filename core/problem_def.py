import pprint
import abc

class Problem(object):
    __metaclass__ = abc.ABCMeta

    def generate_random_arm(self, hps = None):
        if not hps:
            hps = self.hps
        arm = {}
        for hp in hps:
            val = self.domain[hp].get_param_range(1, stochastic=True)
            arm[hp] = val[0]
        return arm

    def print_domain(self):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.domain)

    @abc.abstractmethod
    def _initialise_objective_function(self, x):
        pass

    @abc.abstractmethod
    def _initialise_domain(self):
        pass

