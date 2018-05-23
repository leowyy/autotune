import torch


def get_mean_and_std(dataset):
    """Compute the mean and std value of dataset."""

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(1)
    std = torch.zeros(1)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(1):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def main():
    data_dir = '/Users/signapoop/Desktop/data/'
    from autotune.benchmarks.data.mrbi_data_loader import get_train_val_set, get_test_set

    print('==> Preparing data..')
    train_data, val_data, train_sampler, val_sampler = get_train_val_set(data_dir=data_dir,
                                                                         valid_size=0.2)
    mean, std = get_mean_and_std(train_data)
    print("mean = {}".format(mean))
    print("std = {}".format(std))


if __name__ == "__main__":
    main()
