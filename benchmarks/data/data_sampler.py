import torch
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import matplotlib.pyplot as plt


def get_train_data(problem, data_dir):
    if problem == "mrbi":
        from mrbi_data_loader import get_train_val_set
        train_data, val_data, train_sampler, val_sampler = get_train_val_set(data_dir=data_dir,
                                                                             valid_size=0.2)
    elif problem == "svhn":
        from svhn_data_loader import get_train_val_set
        train_data, val_data, train_sampler, val_sampler = get_train_val_set(data_dir=data_dir,
                                                                             valid_size=0.2)

    elif problem == "cifar":
        from cifar_data_loader import get_train_val_set
        train_data, val_data, train_sampler, val_sampler = get_train_val_set(data_dir=data_dir,
                                                                             augment=False,
                                                                             valid_size=0.2)
    return train_data


def main():
    problem = 'svhn'
    data_dir = '/Users/signapoop/Desktop/data/'

    train_data = get_train_data(problem, data_dir)

    print('Number of samples: ', len(train_data))

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=4,
                                              shuffle=True, num_workers=2)

    # functions to show an image
    def imshow(img):
        # img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))

    # print labels
    print(' '.join('%5s' % labels[j] for j in range(4)))
    plt.show(block=True)


if __name__ == "__main__":
    main()