import argparse
import os
from src.models import models, train_model
from src.scripts.create_dataset import create_dataset
from src.utils import utils
import torch
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='H5N1')
parser.add_argument('--start_year', type=int, default=2001)
parser.add_argument('--end_year', type=int, default=2016)
parser.add_argument('--create_dataset', type=bool, default=False)
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--method', type=str, default='dbscan')
args = parser.parse_args()

parameters = {
    # Exlude _train/_test and file ending
    'data_set': '',

    # 'svm', lstm', 'gru', 'attention' (only temporal) or 'da-rnn' (input and temporal attention)
    'model': 'attention',

    # Number of hidden units in the encoder
    'hidden_size': 128,

    # Droprate (applied at input)
    'dropout_p': 0.5,

    # Note, no learning rate decay implemented
    'learning_rate': 0.001,

    # Size of mini batch
    'batch_size': 256,

    # Number of training iterations
    'num_of_epochs': 50
}

dataset_features = {
    'dataset': args.dataset,

    'num_of_datasets': 5,

    'start_year': args.start_year,

    'end_year': args.end_year,

    'method': args.method
}

if __name__ == '__main__':
    if args.create_dataset:
        for i in range(dataset_features['num_of_runs']):
            create_dataset(dataset_features['start_year'], dataset_features['end_year'], dataset_features['dataset'],
                           i + 1, method=dataset_features['method'])
    if args.train:
        res_path = './results/Tempel/{}_T{}_{}'.format(dataset_features['dataset'],
                                                       dataset_features['end_year'] -
                                                       dataset_features['start_year'],
                                                       dataset_features['end_year'])

        if not os.path.exists(res_path):
            os.mkdir(res_path)
        final_res = {}
        for i in range(5):
            parameters['data_set'] = './data/processed/{}_T{}_{}/{}/triplet_cluster'.format(dataset_features['dataset'],
                                                                                            dataset_features[
                                                                                                'end_year'] -
                                                                                            dataset_features[
                                                                                                'start_year'],
                                                                                            dataset_features[
                                                                                                'end_year'],
                                                                                            i + 1)
            torch.manual_seed(1)
            np.random.seed(1)

            train_trigram_vecs, train_labels = utils.read_dataset(dataset_features['dataset'],
                                                                  parameters['data_set'] + '_train.csv', concat=False)
            test_trigram_vecs, test_labels = utils.read_dataset(dataset_features['dataset'],
                                                                parameters['data_set'] + '_test.csv', concat=False)

            X_train = torch.tensor(train_trigram_vecs, dtype=torch.float32)
            Y_train = torch.tensor(train_labels, dtype=torch.int64)
            X_test = torch.tensor(test_trigram_vecs, dtype=torch.float32)
            Y_test = torch.tensor(test_labels, dtype=torch.int64)

            input_dim = X_train.shape[2]
            seq_length = X_train.shape[0]
            output_dim = 2

            net = models.AttentionModel(seq_length, input_dim, output_dim, parameters['hidden_size'],
                                        parameters['dropout_p'])

            result, (fpr_rnn, tpr_rnn) = train_model.train_rnn(net, False, parameters['num_of_epochs'],
                                                               parameters['learning_rate'],
                                                               parameters['batch_size'], X_train, Y_train, X_test,
                                                               Y_test, False)
            print('Finished')
            df = pd.DataFrame.from_dict(result)
            df.to_csv(res_path + '/{}.csv'.format(i))
            for k, v in result.items():
                if k not in final_res:
                    final_res[k] = [0]
                final_res[k][0] += v[0] / 5

        df = pd.DataFrame.from_dict(final_res)
        df.to_csv(res_path + '/final.csv')

        np.save(res_path + '/fpr', fpr_rnn)
        np.save(res_path + '/tpr', tpr_rnn)
