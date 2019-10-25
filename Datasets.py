from torch.utils.data.sampler import SubsetRandomSampler

import torchvision.transforms as transforms
import torchvision.datasets
import torchvision
import numpy as np
import torch
import pickle
import os

class Datasets():
    def __init__(self):
        self.trainloader = None
        self.valloader = None
        self.testloader = None
        self.dataroot = None


class MNIST(Datasets):
    def __init__(self, dataroot):
        super(MNIST, self).__init__()
        self.dataroot = dataroot + 'MNIST'
        self.inchannels = 1
        self.n_classes = 10

        self.n_training_samples = 10000
        self.n_val_samples = 5000
        self.n_test_samples = 5000

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.train_set = torchvision.datasets.MNIST(root='./data/MNIST', train=True, download=True, transform=transform)
        self.test_set = torchvision.datasets.MNIST(root='./data/MNIST', train=False, download=True, transform=transform)

        self.train_sampler,val_sampler,test_sampler = self.sample_data()

        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=1, sampler=test_sampler, num_workers=2)
        self.val_loader = torch.utils.data.DataLoader(self.train_set, batch_size=64, sampler=val_sampler, num_workers=2)

    def sample_data(self):
        #Training
        train_sampler = SubsetRandomSampler(np.arange(self.n_training_samples, dtype=np.int64))
        #Validation
        val_sampler = SubsetRandomSampler(np.arange(self.n_training_samples, self.n_training_samples + self.n_val_samples, dtype=np.int64))
        #Test
        test_sampler = SubsetRandomSampler(np.arange(self.n_test_samples, dtype=np.int64))
        return train_sampler,val_sampler,test_sampler


    def get_train_loader(self, batch_size):
        train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size,
                                               sampler=self.train_sampler, num_workers=2)
        return(train_loader)




class ImageNet32(Datasets):
    def __init__(self, dataroot):
        self.data_trainroot = dataroot + 'ImageNet/train/'
        self.data_valroot = dataroot + 'ImageNet/val/'
        self.filename = 'train_data_batch_'

        self.train_meanimage = None
        self.train_set = self.load_databatch(data_folder = self.data_trainroot, idx = 1, train = True)
        self.val_loader = self.load_databatch(data_folder = self.data_valroot, idx = 'val_data', train = False)
    def download():
        pass


    def unpickle(self,file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo)
            return dict

    def load_databatch(self,data_folder, idx, img_size=32, train = True):
        if train:
            data_file = os.path.join(data_folder, 'train_data_batch_')
            d = self.unpickle(data_file + str(idx))
            self.train_meanimage = d['mean']
            print(d.keys())

        if not train:
            data_file = os.path.join(data_folder, '')
            d = self.unpickle(data_file + str(idx))
            print(d.keys())

        x = d['data']
        y = d['labels']


        x = x/np.float32(255)
        mean_image = self.train_meanimage /np.float32(255)

        # Labels are indexed from 1, shift it so that indexes start at 0
        y = [i-1 for i in y]
        data_size = x.shape[0]

        x -= mean_image

        img_size2 = img_size * img_size

        x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
        x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

        # create mirrored images
        X_train = x[0:data_size, :, :, :]
        Y_train = y[0:data_size]
        X_train_flip = X_train[:, :, :, ::-1]
        Y_train_flip = (Y_train)
        X_train = torch.tensor(np.concatenate((X_train, X_train_flip), axis=0))
        Y_train = torch.tensor(np.concatenate((Y_train, Y_train_flip), axis=0),dtype=torch.long)

        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        datasets = torch.utils.data.TensorDataset(X_train,Y_train)

        if train:
            return datasets

        dataloader = torch.utils.data.DataLoader(datasets)

        if not train:
            return dataloader



    def get_train_loader(self, batch_size):
        train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size,
                                               sampler=None, num_workers=2)
        return(train_loader)



class CIFAR10(Datasets):

    def __init__(self, dataroot):
        super(CIFAR10, self).__init__()
        self.dataroot = dataroot + 'CIFAR10'
        self.inchannels = 3
        self.n_classes = 10

        self.n_training_samples = 10000
        self.n_val_samples = 5000
        self.n_test_samples = 5000

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

        self.train_set = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=True, download=True, transform=transform_train)
        self.test_set = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=False, download=True, transform=transform_test)


        self.train_sampler,val_sampler,test_sampler = self.sample_data()

        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=128, num_workers=2)
        self.val_loader = torch.utils.data.DataLoader(self.train_set, batch_size=64, num_workers=2)

    def sample_data(self):
        #Training
        train_sampler = SubsetRandomSampler(np.arange(self.n_training_samples, dtype=np.int64))
        #Validation
        val_sampler = SubsetRandomSampler(np.arange(self.n_training_samples, self.n_training_samples + self.n_val_samples, dtype=np.int64))
        #Test
        test_sampler = SubsetRandomSampler(np.arange(self.n_test_samples, dtype=np.int64))
        return train_sampler,val_sampler,test_sampler


    def get_train_loader(self, batch_size):
        train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size,
                                               sampler=self.train_sampler, num_workers=2)
        return(train_loader)
