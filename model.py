import torch
import torch.optim as optim
from sklearn.metrics import roc_curve, auc
import numpy as np
import time

from src.models.train_model import repackage_hidden
from src.utils import validation


class Classifier:
    def __init__(self, lr=0.001, weight_decay=1e-6, lr_milestones=(), n_epochs=50,
                 eps=1e-9, batch_size=128):
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_milestones = lr_milestones
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.eps = eps
        self.scores = {'test': {'precision': [], 'recall': [], 'f-score': [], 'mcc': [], 'accuracy': [], 'auc': []},
                       'valid': {'precision': [], 'recall': [], 'f-score': [], 'mcc': [], 'accuracy': [], 'auc': []}}
        self.roc_info = {}

    def record_results(self, precision, recall, fscore, mcc, acc, auc, roc, dataset):
        self.scores[dataset]['precision'].append(precision)
        self.scores[dataset]['recall'].append(recall)
        self.scores[dataset]['f-score'].append(fscore)
        self.scores[dataset]['mcc'].append(mcc)
        self.scores[dataset]['accuracy'].append(acc)
        self.scores[dataset]['auc'].append(auc)
        if dataset == 'test':
            self.roc_info['fpr'] = roc[0]
            self.roc_info['tpr'] = roc[1]
            self.roc_info['thresh'] = roc[2]

    def test(self, test_loader, net):

        # Testing
        net.eval()
        epoch_loss = 0.0
        n_batches = 0
        idx_label_score = []

        with torch.no_grad():
            for data in test_loader:
                inputs, labels, semi_targets, idx = data
                inputs = inputs.permute(1, 0, 2)

                _, outputs = net(inputs.float(), net.init_hidden(inputs.shape[1]))

                dists = torch.sqrt(torch.norm(outputs, p=2, dim=1) ** 2 + 1) - 1

                scores = 1 - torch.exp(-dists)
                losses = torch.where(semi_targets == 0, dists, -torch.log(scores + self.eps))
                loss = torch.mean(losses)

                # Save triple of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.flatten().cpu().data.numpy().tolist()))

                epoch_loss += loss.item()
                n_batches += 1

        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        fpr, tpr, thresholds = roc_curve(labels, scores)
        test_auc = auc(fpr, tpr)

        return fpr, tpr, thresholds, scores, labels, test_auc, (fpr, tpr, thresholds)

    def eval(self, test_loader, net, thresh=-1):
        fpr, tpr, thresholds, scores, labels, test_auc, roc = self.test(test_loader, net)
        tops = [0, 0]

        if thresh == -1:
            for th in thresholds:
                preds = scores > th
                _, _, fscore, _, _ = validation.evaluate(labels, preds)
                if fscore > tops[0]:
                    tops[0] = fscore
                    tops[1] = th
            preds = scores > tops[1]
        else:
            preds = scores > thresh

        precision, recall, fscore, mcc, val_acc = validation.evaluate(labels, preds)
        self.record_results(precision, recall, fscore, mcc, val_acc, test_auc, roc, 'valid' if thresh == -1 else 'test')

        return tops[1]

    def train(self, net, train_loader, test_loader, valid_loader):

        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)
        for epoch in range(self.n_epochs):

            epoch_loss = 0.0
            n_batches = 0
            hidden = net.init_hidden(self.batch_size)

            for data in train_loader:
                inputs, labels, semi_targets, idx = data
                inputs = inputs.permute(1, 0, 2)
                repackage_hidden(hidden)

                # Zero the network parameter gradients
                if epoch < self.n_epochs:
                    optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs, _ = net(inputs.float(), hidden)

                dists = torch.sqrt(torch.norm(outputs, p=2, dim=1) ** 2 + 1) - 1

                scores = 1 - torch.exp(-dists)
                losses = torch.where(semi_targets == 0, dists, -torch.log(scores + self.eps))
                loss = torch.mean(losses)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # Take learning rate scheduler step
            scheduler.step()

            if self.n_epochs - epoch <= 10:
                thresh = self.eval(valid_loader, net)
                self.eval(test_loader, net, thresh)

        print('Finished training.')

        return net
