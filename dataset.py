import scipy.io
import torch
import numpy as np
import random
import torch.utils.data as data
import copy
import torch.nn as nn

class Dateset_mat():
    def __init__(self, data_path):
        self.img = scipy.io.loadmat(data_path + r"/img.mat")
        self.txt = scipy.io.loadmat(data_path + r"/txt.mat")
        self.label = scipy.io.loadmat(data_path + r"/L.mat")

    def getdata(self):
        self.data = []
        self.data.append(self.img["img"])
        self.data.append(self.txt["txt"])
        self.data.append(self.label["L"])
        fix_seed(1560)
        return self.data

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class data_loder(data.Dataset):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __len__(self):
        return self.data1.__len__()

    def get_data(self, input):
        v1, v2, label = input[0], input[1], input[2]
        size, size1 = v1.__len__(), v2.__len__()

        shuffle_ix = np.random.permutation(np.arange(size))
        v1 = v1[shuffle_ix]
        v2 = v2[shuffle_ix]
        label = label[shuffle_ix]

        # img, txt, label = random.shuffle(img, txt, label)
        assert (size == size1)
        data1, data2, data5 = [], [], []
        alldata1, alldata2, alldata5 = [], [], []
        for i in range(size):
            temp_i = i % self.batch_size
            if temp_i < self.batch_size:
                data1.append(v1[i])
                data2.append(v2[i])
                data5.append(label[i])
            if data1.__len__() == self.batch_size or i == size - 1:
                d1, d2, d5 = copy.deepcopy(data1), copy.deepcopy(data2), copy.deepcopy(data5)
                alldata1.append(d1)
                alldata2.append(d2)
                alldata5.append(d5)

                data1.clear()
                data2.clear()
                data5.clear()
        self.data1 = alldata1
        self.data2 = alldata2
        self.data5 = alldata5

    def __getitem__(self, index):
        v1, v2, label = np.array(self.data1[index]), np.array(self.data2[index]), self.data5[index]
        v1 = torch.tensor(v1, dtype=torch.float32)
        v2 = torch.tensor(v2, dtype=torch.float32)
        label = np.array(label)-1
        label = np.squeeze(label)
        return v1, v2, label

