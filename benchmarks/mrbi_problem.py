from __future__ import division
import torch

from cifar_problem import CifarProblem
from data.mrbi_data_loader import get_train_val_set, get_test_set
from ml_models.cudaconvnet2 import CudaConvNet2


class MrbiProblem(CifarProblem):

    def __init__(self, data_dir, output_dir):
        super(MrbiProblem, self).__init__(data_dir, output_dir)

        # Set this to choose a subset of tunable hyperparams
        # self.hps = None
        self.hps = ['learning_rate', 'n_units_1', 'n_units_2', 'n_units_3', 'batch_size']
        self.name = "MRBI"

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

    def construct_model(self, arm):
        arm['filename'] = arm['dir'] + "/model.pth"

        # Construct model and optimizer based on hyperparameters
        base_lr = arm['learning_rate']
        n_units_1 = int(arm['n_units_1'])
        n_units_2 = int(arm['n_units_2'])
        n_units_3 = int(arm['n_units_3'])
        weight_decay = arm['weight_decay']
        momentum = arm['momentum']

        model = CudaConvNet2(1, n_units_1, n_units_2, n_units_3)  # n_channels = 1
        if self.use_cuda:
            model.cuda()
            model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
            torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
        self.save_checkpoint(arm['filename'], 0, model, optimizer, 1, 1)

        return arm['filename']

