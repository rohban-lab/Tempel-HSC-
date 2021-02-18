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


def load_datasets(dataset, train_path, test_path):
    train_data, train_labels = utils.read_dataset(dataset, train_path, concat=False)
    train_data = np.moveaxis(train_data, [1], [0])
    train_data, valid_data, train_labels, valid_labels = train_test_split(train_data, train_labels, test_size=0.1,
                                                                          random_state=1)
    test_data, test_labels = utils.read_dataset(dataset, test_path, concat=False)

    train_data = np.moveaxis(train_data, [1], [0])
    valid_data = np.moveaxis(valid_data, [1], [0])

    train_dataset = SeqDataset(train_data, train_labels)
    valid_dataset = SeqDataset(valid_data, valid_labels)
    test_dataset = SeqDataset(test_data, test_labels)

    return train_dataset, valid_dataset, test_dataset
