import torch
import torch.nn.functional as F
import math
import time
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from src.utils import utils, validation

def repackage_hidden(h):
  """
  Wraps hidden states in new Tensors, to detach them from their history.
  """
  if isinstance(h, torch.Tensor):
    return h.detach()
  else:
    return tuple(repackage_hidden(v) for v in h)


def plot_training_history(loss, val_loss, acc, val_acc, mini_batch_scores, mini_batch_labels):
  """
  Plots the loss and accuracy for training and validation over epochs.
  Also plots the logits for a small batch over epochs.
  """
  plt.style.use('ggplot')
    
  # Plot losses
  plt.figure()
  plt.subplot(1,3,1)
  plt.plot(loss, 'b', label='Training')
  plt.plot(val_loss, 'r', label='Validation')
  plt.title('Loss')
  plt.legend()

  # Plot accuracies
  plt.subplot(1,3,2)
  plt.plot(acc, 'b', label='Training')
  plt.plot(val_acc, 'r', label='Validation')
  plt.title('Accuracy')
  plt.legend()

  # Plot prediction dynamics of test mini batch
  plt.subplot(1,3,3)
  pos_label, neg_label = False, False
  for i in range(len(mini_batch_labels)):
    if mini_batch_labels[i]:
      score_sequence = [x[i][1] for x in mini_batch_scores]
      if not pos_label:
        plt.plot(score_sequence, 'b', label='Pos')
        pos_label = True
      else:
        plt.plot(score_sequence, 'b')
    else:
      score_sequence = [x[i][0] for x in mini_batch_scores]
      if not neg_label:
        plt.plot(score_sequence, 'r', label='Neg')
        neg_label = True
      else:
        plt.plot(score_sequence, 'r')
  
  plt.title('Logits')
  plt.legend()
  plt.savefig('./reports/figures/training_curves.png')


def plot_attention(weights):
  """
  Plots attention weights in a grid.
  """
  cax = plt.matshow(weights.numpy(), cmap='bone')
  plt.colorbar(cax)
  plt.grid(
    b=False,
    axis='both',
    which='both',
  )
  plt.xlabel('Years')
  plt.ylabel('Examples')
  plt.savefig('./reports/figures/attention_weights.png')


def predictions_from_output(scores):
  """
  Maps logits to class predictions.
  """
  prob = F.softmax(scores, dim=1)
  _, predictions = prob.topk(1)
  return predictions


def verify_model(model, X, Y, batch_size):
  """
  Checks the loss at initialization of the model and asserts that the
  training examples in a batch aren't mixed together by backpropagating.
  """
  print('Sanity checks:')
  criterion = torch.nn.CrossEntropyLoss()
  scores, _ = model(X, model.init_hidden(Y.shape[0]))
  print(' Loss @ init %.3f, expected ~%.3f' % (criterion(scores, Y).item(), -math.log(1 / model.output_dim)))


  mini_batch_X = X[:, :batch_size, :]
  mini_batch_X.requires_grad_()
  criterion = torch.nn.MSELoss()
  scores, _ = model(mini_batch_X, model.init_hidden(batch_size))

  non_zero_idx = 1
  perfect_scores = [[0, 0] for i in range(batch_size)]
  not_perfect_scores = [[1, 1] if i == non_zero_idx else [0, 0] for i in range(batch_size)]

  scores.data = torch.FloatTensor(not_perfect_scores)
  Y_perfect = torch.FloatTensor(perfect_scores)
  loss = criterion(scores, Y_perfect)
  loss.backward()

  zero_tensor = torch.FloatTensor([0] * X.shape[2])
  for i in range(mini_batch_X.shape[0]):
    for j in range(mini_batch_X.shape[1]):
      if sum(mini_batch_X.grad[i, j] != zero_tensor):
        assert j == non_zero_idx, 'Input with loss set to zero has non-zero gradient.'

  mini_batch_X.detach()
  print(' Backpropagated dependencies OK')


def train_rnn(model, verify, epochs, learning_rate, batch_size, X, Y, X_test, Y_test, show_attention):
  """
  Training loop for a model utilizing hidden states.

  verify enables sanity checks of the model.
  epochs decides the number of training iterations.
  learning rate decides how much the weights are updated each iteration.
  batch_size decides how many examples are in each mini batch.
  show_attention decides if attention weights are plotted.
  """
  print_interval = 10
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  criterion = torch.nn.CrossEntropyLoss()
  num_of_examples = X.shape[1]
  num_of_batches = math.floor(num_of_examples/batch_size)

  if verify:
    verify_model(model, X, Y, batch_size)

  all_losses = []
  all_val_losses = []
  all_accs = []
  all_val_accs = []

  # Find mini batch that contains at least one mutation to plot
  plot_batch_size = 10
  i = 0
  while not Y_test[i]:
    i += 1

  X_plot_batch = X_test[:, i:i+plot_batch_size, :]
  Y_plot_batch = Y_test[i:i+plot_batch_size]
  plot_batch_scores = []

  start_time = time.time()
  for epoch in range(epochs):
    model.train()
    running_loss = 0
    running_acc = 0

    hidden = model.init_hidden(batch_size)

    for count in range(0, num_of_examples - batch_size + 1, batch_size):
      repackage_hidden(hidden)

      X_batch = X[:, count:count+batch_size, :]
      Y_batch = Y[count:count+batch_size]

      # print(hidden.shape)
      scores, _ = model(X_batch, hidden)
      loss = criterion(scores, Y_batch)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      predictions = predictions_from_output(scores)
      conf_matrix = validation.get_confusion_matrix(Y_batch, predictions)
      TP, TN = conf_matrix[0][0], conf_matrix[1][1]
      running_acc += TP + TN
      running_loss += loss.item()

    elapsed_time = time.time() - start_time
    epoch_acc = running_acc / Y.shape[0]
    all_accs.append(epoch_acc)
    epoch_loss = running_loss / num_of_batches
    all_losses.append(epoch_loss)

    with torch.no_grad():
      model.eval()
      test_scores, _ = model(X_test, model.init_hidden(Y_test.shape[0]))
      predictions = predictions_from_output(test_scores)
      predictions = predictions.view_as(Y_test)
      
      precision, recall, fscore, mcc, val_acc = validation.evaluate(Y_test, predictions)

      val_loss = criterion(test_scores, Y_test).item()
      all_val_losses.append(val_loss)
      all_val_accs.append(val_acc)

      plot_scores, _ = model(X_plot_batch, model.init_hidden(Y_plot_batch.shape[0]))
      plot_batch_scores.append(plot_scores)

    if epoch % print_interval == 0:
      print(' Epoch %d\tTime %s\tT_loss %.3f\tT_acc  %.3f\tV_loss %.3f\tV_acc  %.3f\tPrecis %.3f\tRecall %.3f\tFscore %.3f\tMCC    %.3f'
        % (epoch, utils.get_time_string(elapsed_time), epoch_loss, epoch_acc, val_loss, val_acc, precision, recall, fscore, mcc))

  plot_training_history(all_losses, all_val_losses, all_accs, all_val_accs, plot_batch_scores, Y_plot_batch)
  if show_attention:
    with torch.no_grad():
      model.eval()
      _, attn_weights = model(X_plot_batch, model.init_hidden(Y_plot_batch.shape[0]))
      plot_attention(attn_weights)
  plt.show()


def svm_baseline(X, Y, X_test, Y_test, method=None):
    clf = SVC(gamma='auto', class_weight='balanced').fit(X, Y) 
    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = validation.evaluate(Y_test, Y_pred)
    print('SVM baseline:')
    print('V_acc  %.3f\tPrecis %.3f\tRecall %.3f\tFscore %.3f\tMCC %.3f'
                % (val_acc, precision, recall, fscore, mcc))
    if(method!=None):
        with open(f'./reports/results/{method}_SVM.txt', 'a') as f:
            f.write(' Accuracy:\t%.3f\n' % val_acc)
            f.write(' Precision:\t%.3f\n' % precision)
            f.write(' Recall:\t%.3f\n' % recall)
            f.write(' F1-score:\t%.3f\n' % fscore)
            f.write(' Matthews CC:\t%.3f\n\n' % mcc)
