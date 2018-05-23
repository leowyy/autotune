from __future__ import division
import torch

from cifar_problem import CifarProblem
from data.mrbi_data_loader import get_train_val_set, get_test_set


class MrbiProblem(CifarProblem):

    def __init__(self, data_dir, output_dir):
        super(MrbiProblem, self).__init__(data_dir, output_dir)

        # Set this to choose a subset of tunable hyperparams
        self.hps = None
        # self.hps = ['learning_rate', 'n_units_1', 'n_units_2', 'n_units_3', 'batch_size']

    def initialise_data(self):
        # 9.6k train, 2.4k val, 50k test
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

