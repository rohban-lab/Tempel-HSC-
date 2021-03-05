from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np

from src.utils import utils


class SeqDataset(Dataset):

    def __init__(self, data, label):
        super(Dataset, self).__init__()

        self.data = data
        self.labels = label

    def __getitem__(self, index):
        return self.data[:, index, :], int(self.labels[index]), int(self.labels[index]), index

    def __len__(self):
        return self.data.shape[1]


def load_datasets(dataset, train_path, test_path, all_years_label=False):
    train_data, train_labels = utils.read_dataset(dataset, train_path, concat=False, all_years_label=all_years_label)

    length = int(train_data.shape[1] / 10)
    valid_data = train_data[:, 0:length, :]
    valid_labels = train_labels[0:length]
    train_data = train_data[:, length:, :]
    train_labels = train_labels[length:]

    test_data, test_labels = utils.read_dataset(dataset, test_path, concat=False, all_years_label=all_years_label)

    # length = int(test_data.shape[1]/10)
    # valid_data = test_data[:,0:length,:]
    # valid_labels = test_labels[0:length]
    # test_data = test_data[:,length:,:]
    # test_labels = test_labels[length:]

    train_dataset = SeqDataset(train_data, train_labels)
    valid_dataset = SeqDataset(valid_data, valid_labels)
    test_dataset = SeqDataset(test_data, test_labels)

    return train_dataset, valid_dataset, test_dataset
