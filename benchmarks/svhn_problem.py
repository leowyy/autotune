from __future__ import division
import torch

from cifar_problem import CifarProblem
from ..core.params import *
from data.svhn_data_loader import get_train_val_set, get_test_set


class SvhnProblem(CifarProblem):

    def __init__(self, data_dir, output_dir):
        super(SvhnProblem, self).__init__(data_dir, output_dir)

        # Set this to choose a subset of tunable hyperparams
        # self.hps = None
        self.hps = ['learning_rate', 'n_units_1', 'n_units_2', 'n_units_3', 'batch_size']

    def initialise_data(self):
        # 40k train, 10k val, 10k test
        print('==> Preparing data..')
        train_data, val_data, train_sampler, val_sampler = get_train_val_set(data_dir=self.data_dir,
                                                                             valid_size=0.2)
        test_data = get_test_set(data_dir=self.data_dir)

        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=100, sampler=val_sampler,
                                                      num_workers=2, pin_memory=False)
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True,
                                                       num_workers=2, pin_memory=False)
        self.train_data = train_data
        self.train_sampler = train_sampler

    def initialise_domain(self):
        params = {
            'learning_rate': Param('learning_rate', -6, 0, distrib='uniform', scale='log', logbase=10),
            'n_units_1': Param('n_units_1', 4, 8, distrib='uniform', scale='log', logbase=2, interval=1),
            'n_units_2': Param('n_units_2', 4, 8, distrib='uniform', scale='log', logbase=2, interval=1),
            'n_units_3': Param('n_units_3', 4, 8, distrib='uniform', scale='log', logbase=2, interval=1),
            'batch_size': Param('batch_size', 32, 512, distrib='uniform', scale='linear', interval=1),
            'lr_step': Param('lr_step', 1, 5, distrib='uniform', init_val=1, scale='linear', interval=1),
            'gamma': Param('gamma', -3, -1, distrib='uniform', init_val=0.1, scale='log', logbase=10),
            'weight_decay': Param('weight_decay', -6, -1, init_val=0.004, distrib='uniform', scale='log', logbase=10),
            'momentum': Param('momentum', 0.3, 0.999, init_val=0.9, distrib='uniform', scale='linear'),
        }
        return params
