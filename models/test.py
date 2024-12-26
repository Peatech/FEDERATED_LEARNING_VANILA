#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

"""
This test_img function is used to evaluate the performance of the global model (net_g) 
on a given test dataset (datatest). It calculates the test loss and accuracy of 
the model and optionally prints the results if args.verbose is set.
"""
def test_img(net_g, datatest, args):
    net_g.eval()  # sets the model (net_g) to evaluation mode
    # testing
    test_loss = 0 # Accumulates the total loss over all test samples.
    correct = 0  # Counts the number of correctly predicted labels.
    data_loader = DataLoader(datatest, batch_size=args.bs)  # Creates the dataloader with batch size of args.bs
    l = len(data_loader)

    #  Iterating Through Test Data. This loops through the batches of test data
    for idx, (data, target) in enumerate(data_loader):
        # Ensures the data and labels are moved to the correct device (CPU or GPU):
        if torch.cuda.is_available() and args.gpu != -1:
            data, target = data.cuda(args.device), target.cuda(args.device)
        else:
            data, target = data.cpu(), target.cpu()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss
