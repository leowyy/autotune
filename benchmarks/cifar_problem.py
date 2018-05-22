from __future__ import division
import numpy as np
import torch
import torch.nn as nn

from torch_net_problem import TorchNetProblem
from ..core.params import *
from data.cifar_data_loader import get_train_val_set, get_test_set
from ml_models.cudaconvnet2 import CudaConvNet2


class CifarProblem(TorchNetProblem):

    def __init__(self, data_dir, output_dir):
        super(CifarProblem, self).__init__(data_dir, output_dir)

        # Set this to choose a subset of tunable hyperparams
        # self.hps = None
        self.hps = ['learning_rate', 'n_units_1', 'n_units_2', 'n_units_3', 'batch_size']

    def initialise_data(self):
        # 40k train, 10k val, 10k test
        print('==> Preparing data..')
        train_data, val_data, train_sampler, val_sampler = get_train_val_set(data_dir=self.data_dir,
                                                                             augment=True,
                                                                             valid_size=0.2)
        test_data = get_test_set(data_dir=self.data_dir)

        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=100, sampler=val_sampler,
                                                      num_workers=2, pin_memory=False)
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True,
                                                       num_workers=2, pin_memory=False)
        self.train_data = train_data
        self.train_sampler = train_sampler

    def construct_model(self, arm):
        arm['filename'] = arm['dir'] + "/model.pth"

        # Construct model and optimizer based on hyperparameters
        base_lr = arm['learning_rate']
        n_units_1 = int(arm['n_units_1'])
        n_units_2 = int(arm['n_units_2'])
        n_units_3 = int(arm['n_units_3'])
        weight_decay = arm['weight_decay']
        momentum = arm['momentum']

        model = CudaConvNet2(n_units_1, n_units_2, n_units_3)
        if self.use_cuda:
            model.cuda()
            model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
            torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
        self.save_checkpoint(arm['filename'], 0, model, optimizer, 1, 1)

        return arm['filename']

    def eval_arm(self, arm, n_resources):
        print("\nLoading arm with parameters.....")

        arm['n_resources'] = arm['n_resources'] + n_resources
        print(arm)

        # Load model and optimiser from file to resume training
        checkpoint = torch.load(arm['filename'])
        start_epoch = checkpoint['epoch']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

        # Rest of the tunable hyperparameters
        base_lr = arm['learning_rate']
        batch_size = int(arm['batch_size'])
        lr_step = int(arm['lr_step'])
        gamma = arm['gamma']

        # Initialise train_loader based on batch size
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size,
                                                   sampler=self.train_sampler,
                                                   num_workers=2, pin_memory=False)

        # Compute derived hyperparameters
        n_batches = int(n_resources * 10000 / batch_size)  # each unit of resource = 10,000 examples
        batches_per_epoch = len(train_loader)
        max_epochs = int(n_batches / batches_per_epoch) + 1

        if lr_step > max_epochs:
            step_size = max_epochs
        else:
            step_size = int(max_epochs / lr_step)

        criterion = nn.CrossEntropyLoss()

        for epoch in range(start_epoch, start_epoch+max_epochs):
            # Adjust learning rate by decay schedule
            self.adjust_learning_rate(optimizer, epoch, base_lr, gamma, step_size)

            # Train the net for one epoch
            self.train(train_loader, model, optimizer, criterion, epoch, min(n_batches, batches_per_epoch))

            # Decrement n_batches remaining
            n_batches = n_batches - batches_per_epoch

        # Evaluate trained net on val and test set
        val_error = self.test(self.val_loader, model, criterion)
        test_error = self.test(self.test_loader, model, criterion)

        self.save_checkpoint(arm['filename'], start_epoch+max_epochs, model, optimizer, val_error, test_error)

        return val_error, test_error

    def initialise_domain(self):
        params = {
            'learning_rate': Param('learning_rate', np.log(10**-6), np.log(10**0), distrib='uniform', scale='log'),
            'n_units_1': Param('n_units_1', np.log(2**4), np.log(2**8), distrib='uniform', scale='log', interval=1),
            'n_units_2': Param('n_units_2', np.log(2**4), np.log(2**8), distrib='uniform', scale='log', interval=1),
            'n_units_3': Param('n_units_3', np.log(2**4), np.log(2**8), distrib='uniform', scale='log', interval=1),
            'batch_size': Param('batch_size', 32, 512, distrib='uniform', scale='linear', interval=1),
            'lr_step': Param('lr_step', 1, 5, distrib='uniform', init_val=1, scale='linear', interval=1),
            'gamma': Param('gamma', np.log(10**-3), np.log(10**-1), distrib='uniform', init_val=0.1, scale='log'),
            'weight_decay': Param('weight_decay', np.log(10**-6), np.log(10**-1), init_val=0.004, distrib='uniform', scale='log'),
            'momentum': Param('momentum', 0.3, 0.999, init_val=0.9, distrib='uniform', scale='linear'),
        }
        return params
