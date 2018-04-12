from __future__ import division
import os
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from problem_def import Problem
from params import Param
from utils import progress_bar
from ..sandpit.models.cudaconvnet import *

class CIFAR10_problem(Problem):
    def __init__(self, data_dir, dirname):
        self.dirname = dirname
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self.data_dir = data_dir
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        self.trainloader, self.testloader = self._initialise_data()
        self.eval_arm = lambda x: self._initialise_objective_function(x)
        self.domain = self._initialise_domain()
        self.hps = ['learning_rate', 'scale', 'power', 'lr_step']

        self.use_cuda = torch.cuda.is_available()
        print("Using GPUs? :", self.use_cuda)

    def _initialise_data(self):
        # TODO: confirm if the following data preprocessing is necessary
        # TODO: train, val, test split

        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True,
                                                transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

        return trainloader, testloader

    def _initialise_objective_function(self, arm):
        print(arm)

        # Tunable hyperparameters
        n_units = arm['n_units']
        base_lr = arm['learning_rate']
        lr_step = arm['lr_step']
        scale   = arm['scale']
        power   = arm['power']

        # Default hyperparameters
        n_batches = n_units * 100 # each unit of resource = 100 mini batches
        max_epochs = int(n_batches/500) + 1
        l2_flag = False
        gamma = 0.1
        if lr_step > max_epochs or lr_step == 0:
            step_size = max_epochs
        else:
            step_size = int(max_epochs / lr_step)

        model = CudaConvNet(scale, power)
        if self.use_cuda:
            model.cuda()
            model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.004)

        def l2_penalty(var):
            return torch.sqrt(torch.pow(var, 2).sum())

        def adjust_learning_rate(optimizer, epoch):
            """Sets the learning rate to the initial LR decayed by gamma every 'step_size' epochs"""
            lr = base_lr * (gamma ** (epoch // step_size))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Training
        def train(epoch, max_batches = 500):
            print('\nEpoch: %d' % epoch)
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            adjust_learning_rate(optimizer, epoch)

            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                if self.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                if batch_idx >= max_batches: break
                optimizer.zero_grad()
                inputs, targets = Variable(inputs), Variable(targets)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # if l2_flag:
                #     l2_reg = Variable(torch.FloatTensor(1), requires_grad=True)
                #     # Add l2 penalty during training
                #     for i, W in enumerate(model.parameters()):
                #         q, r = divmod(i, 2)
                #         if r == 0 and q <= 3:
                #             l2_reg = l2_reg + float(weight_costs[q]) * torch.pow(W, 2).sum()
                #     loss = loss + l2_reg

                loss.backward()
                optimizer.step()

                train_loss += loss.data[0]
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

                progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            return train_loss

        def test(epoch):
            global best_acc
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                if self.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = Variable(inputs, volatile=True), Variable(targets)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.data[0]
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

                progress_bar(batch_idx, len(self.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

            # Save checkpoint.
            test_acc = correct / total
            return test_acc

        best_acc = 0
        val_acc = 0
        filename = self.dirname + 'checkpoint.pth.tar'
        for epoch in range(max_epochs):
            train(epoch, min(n_batches, 500))
            n_batches = n_batches - 500
            test_acc = test(epoch)

            if test_acc > best_acc:
                print('Saving..')
                state = {
                    'model': model.module if self.use_cuda else model,
                    'test_acc': test_acc,
                    'epoch': epoch,
                }
                torch.save(state, filename)
                best_acc = test_acc

        return 1-test_acc

    def _initialise_domain(self):
        params = {}
        params['learning_rate'] = Param('learning_rate', np.log(5e-5), np.log(5), distrib='uniform', scale='log')
        params['weight_cost1'] = Param('weight_cost1', np.log(5e-5), np.log(5), distrib='uniform', scale='log')
        params['weight_cost2'] = Param('weight_cost2', np.log(5e-5), np.log(5), distrib='uniform', scale='log')
        params['weight_cost3'] = Param('weight_cost3', np.log(5e-5), np.log(5), distrib='uniform', scale='log')
        params['weight_cost4'] = Param('weight_cost4', np.log(5e-3), np.log(500), distrib='uniform', scale='log')
        params['scale'] = Param('scale', np.log(5e-6), np.log(5), distrib='uniform', scale='log')
        params['power'] = Param('power', 0.01, 3, distrib='uniform', scale='linear')
        params['lr_step'] = Param('lr_step', 1, 5, distrib='uniform', scale='linear', interval=1)

        return params