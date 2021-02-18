from src.models import models, train_model
from src.data import make_dataset
from src.features import build_features
from src.utils import utils
import torch
import numpy as np
import operator


parameters = {
  # Exlude _train/_test and file ending
  'data_set': './data/processed/triplet_dbscan',

  # 'svm', lstm', 'gru', 'attention' (only temporal) or 'da-rnn' (input and temporal attention)
  'model': 'attention',

  # Number of hidden units in the encoder
  'hidden_size': 128,

  # Droprate (applied at input)
  'dropout_p': 0.1,

  # Note, no learning rate decay implemented
  'learning_rate': 0.001,

  # Size of mini batch
  'batch_size': 512,

  # Number of training iterations
  'num_of_epochs': 50
}

torch.manual_seed(1)
np.random.seed(1)

train_trigram_vecs, train_labels = utils.read_dataset('H5N1', parameters['data_set'] + '_train.csv', concat=False)
test_trigram_vecs, test_labels = utils.read_dataset('H5N1', parameters['data_set'] + '_test.csv', concat=False)

X_train = torch.tensor(train_trigram_vecs, dtype=torch.float32)
Y_train = torch.tensor(train_labels, dtype=torch.int64)
X_test = torch.tensor(test_trigram_vecs, dtype=torch.float32)
Y_test = torch.tensor(test_labels, dtype=torch.int64)

_, counts = np.unique(Y_train, return_counts=True)
imbalance = max(counts) / Y_train.shape[0]
print('Class imbalances:')
print(' Training %.3f' % imbalance)
_, counts = np.unique(Y_test, return_counts=True)
imbalance = max(counts) / Y_test.shape[0]
print(' Testing  %.3f' % imbalance)
with open(parameters['data_set'] + '_test_baseline.txt', 'r') as f:
  print('Test baselines:')
  print(f.read())

if parameters['model'] == 'svm':
    window_size = 1
    train_model.svm_baseline(
        build_features.reshape_to_linear(train_trigram_vecs, window_size=window_size), train_labels, 
        build_features.reshape_to_linear(test_trigram_vecs, window_size=window_size), test_labels)
else:
  print(X_train.shape)
  input_dim = X_train.shape[2]
  seq_length = X_train.shape[0]
  output_dim = 2

  if parameters['model'] == 'lstm':
    net = models.RnnModel(input_dim, output_dim, parameters['hidden_size'], parameters['dropout_p'], cell_type='LSTM')
  elif parameters['model'] == 'gru':
    net = models.RnnModel(input_dim, output_dim, parameters['hidden_size'], parameters['dropout_p'], cell_type='LSTM')
  elif parameters['model'] == 'attention':
    net = models.AttentionModel(seq_length, input_dim, output_dim, parameters['hidden_size'], parameters['dropout_p'])
  elif parameters['model'] == 'da-rnn':
    net = models.DaRnnModel(seq_length, input_dim, output_dim, parameters['hidden_size'], parameters['dropout_p'])

  train_model.train_rnn(net, False, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], X_train, Y_train, X_test, Y_test, True)
