
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics

"""
This is a wrapper around the standard PyTorch dataset. Its purpose is to allow each client to only access its own subset of the dataset.
Initialization (__init__ method):
dataset: The full dataset (e.g., MNIST, CIFAR10).
idxs: Indices of the dataset samples that belong to a particular client.
Example:

idxs = [0, 1, 2, ..., 599]  # Client 0 owns these samples
client_dataset = DatasetSplit(full_dataset, idxs)
Length (__len__ method):
Returns the number of samples owned by the client.
Item Retrieval (__getitem__ method):
Given an index (item), it retrieves the corresponding image and label from the dataset.
"""
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

"""
This simulates local training on a client using its own dataset.
Initialization (__init__)
Inputs:
args: Contains various configurations like batch size, learning rate, device type (CPU/GPU), etc.
dataset: The full dataset.
idxs: Indices of samples that belong to this client.
Purpose:
Creates a DataLoader (ldr_train) for the client's data to enable mini-batch training.
Sets the loss function (CrossEntropyLoss), learning rate, and learning rate decay.

This method handles local training for a single client.

Setup:
Sets the model (net) to training mode.
Initializes the optimizer (SGD with momentum) and learning rate scheduler.
Training Loop:
Outer Loop (for iter in range(self.args.local_ep)):
Runs for the specified number of local epochs (local_ep).
Inner Loop (Mini-batch Training):
Retrieves a batch of images and labels from the client's data (self.ldr_train).
Performs the following steps for each mini-batch:
Forward Pass:
Computes predictions (log_probs) using the model.
Loss Calculation:
Computes the error (loss) between predictions and true labels using CrossEntropyLoss.
Backward Pass:
Computes gradients of the loss with respect to the model’s parameters.
Parameter Update:
Updates the model's parameters using the optimizer.
Learning Rate Adjustment:
Adjusts the learning rate after each mini-batch using the scheduler.
Log Loss (Optional):
If verbose is enabled, prints loss information every 10 mini-batches.
Output:
After training, the method returns:
net.state_dict(): The updated model weights after training.
sum(epoch_loss) / len(epoch_loss): The average loss over all epochs.
"""
class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.lr = args.lr
        self.lr_decay = args.lr_decay

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=self.args.momentum)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.lr_decay)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                # print(list(log_probs.size()))
                # print(labels)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        self.lr = scheduler.get_last_lr()[0]
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
