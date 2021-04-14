import torch as tc
import torchvision as tv
import numpy as np


class CIFAR10ZagoruykoPreprocessing(tc.utils.data.Dataset):
    # CIFAR-10 with preprocessing as described in Section 3 of Zagoruyko and Komodakis, 2016.
    def __init__(self, root, train):
        self.root = root
        self.train = train
        self.dataset = tv.datasets.CIFAR10(
            root=root, train=train, download=True, transform=None, target_transform=None)
        self.per_channel_means = self.get_per_channel_means(self.root)
        self.per_channel_std = self.get_per_channel_std(self.root, self.per_channel_means)
        self.transform = self.get_transform(self.train, self.per_channel_means, self.per_channel_std)
        self.target_transform = None

    @staticmethod
    def get_per_channel_means(root):
        training_data = tv.datasets.CIFAR10(
            root=root, train=True, download=True, transform=None)

        per_channel_means = np.zeros(dtype=np.float32, shape=(3,))

        for i in range(0, len(training_data)):
            X, y = training_data[i]
            X = np.array(X).astype(np.float32)
            X = X / 255. # convert to [0, 1] range
            per_channel_means += np.mean(X, axis=(0,1)) # per-channel mean, in NHWC format.

        per_channel_means = per_channel_means / float(len(training_data))
        return per_channel_means

    @staticmethod
    def get_per_channel_std(root, per_channel_means):
        training_data = tv.datasets.CIFAR10(
            root=root, train=True, download=True, transform=None)

        per_channel_variances = np.zeros(dtype=np.float32, shape=(3,))

        for i in range(0, len(training_data)):
            X, y = training_data[i]
            X = np.array(X).astype(np.float32)
            X = X / 255. # convert to [0, 1] range
            per_channel_variances += np.mean(np.square(X-per_channel_means), axis=(0,1)) # NHWC format.

        per_channel_variances = per_channel_variances / float(len(training_data))
        per_channel_stddevs = np.sqrt(per_channel_variances + 1e-6)
        return per_channel_stddevs

    @staticmethod
    def get_transform(train, per_channel_means, per_channel_std):
        if train:
            return tv.transforms.Compose([
                tv.transforms.RandomHorizontalFlip(p=0.5),
                tv.transforms.RandomCrop(size=(32, 32), padding=4, padding_mode='symmetric'),
                tv.transforms.ToTensor(), # [0,1] range, NCHW format.
                tv.transforms.Normalize(mean=per_channel_means, std=per_channel_std)
            ])
        else:
            return tv.transforms.Compose([
                tv.transforms.ToTensor(), # [0,1] range, NCHW format.
                tv.transforms.Normalize(mean=per_channel_means, std=per_channel_std)
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        X, y = self.dataset[idx]

        image = self.transform(X)
        label = y

        return image, label
