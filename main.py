
import random
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os



if __name__ == '__main__':

  """
  The essence of these lines of code is to ensure reproducibility in your experiments.
  By setting fixed seeds for random number generators, you ensure that:
  The same random sequences are generated across multiple runs of your code.
  """
  random.seed(123)
  np.random.seed(123)
  torch.manual_seed(123)
  torch.cuda.manual_seed_all(123)
  torch.cuda.manual_seed(123)

  # Parse Arguments
  args = args_parser()

  # Set Device
  args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


###########################################
  # SET 1: LOAD DATASET and SPLIT USERS
###########################################
  """
  The dataset is loaded and preprocessed using normalization and transformation (e.g., converting images to tensors)
  transforms.Normalize((mean,), (std,)): Normalizes the dataset by subtracting the mean and dividing by the standard deviation (std) for each channel.
  E.g in MNIST:
  Mean: 0.1307 (the average pixel intensity across the MNIST dataset).
  Std: 0.3081 (the standard deviation of pixel intensities across the dataset).
  """
  if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)

  
