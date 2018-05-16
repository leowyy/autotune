from __future__ import division
import numpy as np
import torch
import torch.nn as nn

from torch_net_problem import TorchNetProblem
from ..core.params import *
from data.mnist_data_loader import get_train_val_set, get_test_set
from ml_models.logistic_regression import LogisticRegression


class MnistProblem(TorchNetProblem):

    def __init__(self, data_dir, output_dir):
        super(MnistProblem, self).__init__(data_dir, output_dir)
        self.hps = None

    def initialise_data(self):
        # 48k train, 12k val, 10k test
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

    def construct_model(self, arm):
        arm['filename'] = arm['dir'] + "/model.pth"

        # Construct model and optimizer based on hyperparameters
        base_lr = arm['learning_rate']
        momentum = arm['momentum']
        weight_decay = arm['weight_decay']

        input_size = 784
        num_classes = 10
        model = LogisticRegression(input_size, num_classes)
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
        batch_size = int(arm['batch_size'])

        # Initialise train_loader based on batch size
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size,
                                                   sampler=self.train_sampler,
                                                   num_workers=2, pin_memory=False)

        # Compute derived hyperparameters
        n_batches = int(n_resources * 10000 / batch_size)  # each unit of resource = 10,000 examples
        batches_per_epoch = len(train_loader)
        max_epochs = int(n_batches / batches_per_epoch) + 1

        criterion = nn.CrossEntropyLoss()

        for epoch in range(start_epoch, start_epoch+max_epochs):
            # Train the net for one epoch
            self.train(train_loader, model, optimizer, criterion, epoch, min(n_batches, batches_per_epoch))

            # Decrement n_batches remaining
            n_batches = n_batches - batches_per_epoch

        # Evaluate trained net on val and test set
        val_error = self.test(self.val_loader, model, criterion)
        test_error = self.test(self.test_loader, model, criterion)

        self.save_checkpoint(arm['filename'], epoch, model, optimizer, val_error, test_error)

        return val_error, test_error

    def initialise_domain(self):
        params = {
            'learning_rate': Param('learning_rate', np.log(10**-6), np.log(10**0), distrib='uniform', scale='log'),
            'weight_decay': Param('weight_decay', np.log(10**-6), np.log(10**-1), distrib='uniform', scale='log'),
            'momentum': Param('momentum', 0.3, 0.999, distrib='uniform', scale='linear'),
            'batch_size': Param('batch_size', 20, 2000, distrib='uniform', scale='linear', interval=1),
        }
        return params
