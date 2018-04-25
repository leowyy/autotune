from __future__ import division
import os
import abc
import torch
from torch.autograd import Variable
from ..core.problem_def import Problem

from ..util.progress_bar import progress_bar


class TorchNetProblem(Problem):

    __metaclass__ = abc.ABCMeta

    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.initialise_data()
        self.domain = self.initialise_domain()

        self.use_cuda = torch.cuda.is_available()
        print("Using GPUs? : {}".format(self.use_cuda))

    def train(self, loader, model, optimizer, criterion, epoch, max_batches, disp_interval=500):
        print('\nEpoch: %d' % epoch)
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(loader, start=1):
            if batch_idx >= max_batches:
                break

            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            if batch_idx % disp_interval == 0 or batch_idx == len(loader):
                progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (train_loss / batch_idx, 100. * correct / total, correct, total))
        return train_loss

    def test(self, loader, model, criterion, disp_interval=100):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(loader, start=1):
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            if batch_idx % disp_interval == 0 or batch_idx == len(loader):
                progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / batch_idx, 100. * correct / total, correct, total))

        return 1 - correct / total

    def save_checkpoint(self, filename, epoch, model, optimizer, val_error, test_error):
        torch.save({
            'epoch': epoch,
            'model': model,
            'optimizer': optimizer,
            'val_error': val_error,
            'test_error': test_error,
        }, filename)
        return filename

    def adjust_learning_rate(self, optimizer, epoch, base_lr, gamma, step_size):
        """Sets the learning rate to the initial LR decayed by gamma every step_size epochs"""
        lr = base_lr * (gamma ** (epoch // step_size))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def generate_arms(self, n, hps=None):
        arms = super(TorchNetProblem, self).generate_arms(n, hps)
        os.chdir(self.output_dir)

        subdirs = next(os.walk('.'))[1]
        if len(subdirs) == 0:
            start_count = 0
        else:
            start_count = len(subdirs)

        for i in range(n):
            dirname = "arm" + str(start_count+i)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            arms[i]['n_resources'] = 0
            arms[i]['dir'] = self.output_dir + dirname
            self.construct_model(arms[i])
        return arms

    @abc.abstractmethod
    def initialise_data(self):
        pass

    @abc.abstractmethod
    def construct_model(self, arm):
        pass
