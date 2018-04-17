import torch.nn as nn
import torch.nn.functional as F

class CudaConvNet(nn.Module):
    def __init__(self, alpha=0.00005, beta=0.010001):
        super(CudaConvNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 5, 1, 2)  # n_input, n_output, ks, stride, padding
        self.conv2 = nn.Conv2d(32, 32, 5, 1, 2)
        self.conv3 = nn.Conv2d(32, 64, 5, 1, 2)

        self.lrn1 = LRN(3, alpha, beta)  # local size = 3
        self.lrn2 = LRN(3, alpha, beta)

        self.fc1 = nn.Linear(64 * 4 * 4, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(F.max_pool2d(out, 3, 2, 1), inplace=True)  # ks, stride, padding
        out = self.lrn1(out)

        out = F.relu(self.conv2(out), inplace=True)
        out = F.avg_pool2d(out, 3, 2, 1)
        out = self.lrn2(out)

        out = F.relu(self.conv3(out), inplace=True)
        out = F.avg_pool2d(out, 3, 2, 1)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        return out

# Helper class for local response normalisation
# Is this efficient?
class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta


    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x
