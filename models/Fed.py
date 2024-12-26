#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn

"""
The FedWeightAvg function implements Federated Averaging (FedAvg) for updating global 
model weights in a Federated Learning (FL) setting. This function aggregates the weights (w) of 
local models trained on different clients using a weighted average, where the weights are proportional to the size of each client's dataset.
w: A list of state dictionaries (weights of local models) from different clients.
size: A list of dataset sizes corresponding to each client.
w_avg: A single state dictionary containing the globally averaged weights.
We copy w[0] alone in the line:
w_avg = copy.deepcopy(w[0])
By copying w[0], we initialize w_avg with the correct structure (e.g., all layers, weights, and biases) required for the global model.
Without copying w[0], we would have to manually initialize w_avg with a structure that matches the neural network, which is prone to errors and less efficient.
copy.deepcopy(w[0]) ensures that changes made to w_avg during the aggregation process do not affect the original weights of w[0] (or any other client's weights).

For each parameter (k), it adds the weighted contribution from every other client's model:
w_avg [k] = w_avg [k]+(w[i][k]×size[i])

Example:
w = [
    {"layer1.weight": torch.tensor([1.0, 2.0]), "layer1.bias": torch.tensor([0.5])},  # Client 1
    {"layer1.weight": torch.tensor([2.0, 3.0]), "layer1.bias": torch.tensor([1.5])},  # Client 2
]
size = [3, 7]  # Client 1 has 3 samples, Client 2 has 7 samples
Steps:

Total size: totalSize = 3 + 7 = 10.
Initialize w_avg as a deep copy of the first client's weights:
w_avg = {"layer1.weight": torch.tensor([1.0, 2.0]), "layer1.bias": torch.tensor([0.5])}
Scale the first client's weights:
w_avg["layer1.weight"] = torch.tensor([1.0, 2.0]) * 3  # [3.0, 6.0]
w_avg["layer1.bias"] = torch.tensor([0.5]) * 3  # [1.5]
Add the second client's weighted contribution:
w_avg["layer1.weight"] += torch.tensor([2.0, 3.0]) * 7  # [3.0 + 14.0, 6.0 + 21.0] = [17.0, 27.0]
w_avg["layer1.bias"] += torch.tensor([1.5]) * 7  # [1.5 + 10.5] = [12.0]
Normalize:
w_avg["layer1.weight"] = torch.div(w_avg["layer1.weight"], 10)  # [1.7, 2.7]
w_avg["layer1.bias"] = torch.div(w_avg["layer1.bias"], 10)  # [1.2]
Output:
w_avg = {"layer1.weight": torch.tensor([1.7, 2.7]), "layer1.bias": torch.tensor([1.2])}

"""
def FedWeightAvg(w, size):
    totalSize = sum(size)  # Adds up the number of data samples across all clients to get the total dataset size (totalSize).
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w[0][k]*size[0]
    for k in w_avg.keys():  # This loop iterates through all the clients except the first.
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * size[i]
        # print(w_avg[k])
        w_avg[k] = torch.div(w_avg[k], totalSize)  # Normalize the Averaged Weights. After aggregating all the weighted contributions for a parameter, divide the sum by the total dataset size (totalSize).
    return w_avg
