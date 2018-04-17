import torch.nn as nn
import torch.nn.functional as F

class CudaConvNet2(nn.Module):
    def __init__(self, n_units_1, n_units_2, n_units_3):
        super(CudaConvNet2, self).__init__()

        self.conv1 = nn.Conv2d(3, n_units_1, 5, 1, 2)  # n_input, n_output, ks, stride, padding
        self.conv2 = nn.Conv2d(n_units_1, n_units_2, 5, 1, 2)
        self.conv3 = nn.Conv2d(n_units_2, n_units_3, 5, 1, 2)

        self.norm1 = nn.BatchNorm1d(n_units_1)
        self.norm2 = nn.BatchNorm1d(n_units_2)
        self.norm3 = nn.BatchNorm1d(n_units_3)

        self.fc1 = nn.Linear(n_units_3 * 4 * 4, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = self.norm1(out)
        out = F.max_pool2d(out, 3, 2, 1)  # ks, stride, padding

        out = F.relu(self.conv2(out), inplace=True)
        out = self.norm2(out)
        out = F.max_pool2d(out, 3, 2, 1)

        out = F.relu(self.conv3(out), inplace=True)
        out = self.norm3(out)
        out = F.max_pool2d(out, 3, 2, 1)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        return out
