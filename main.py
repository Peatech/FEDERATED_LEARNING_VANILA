
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


######################################################################################
  # SET 1: LOAD DATASET and SPLIT USERS
######################################################################################
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
            dict_users = mnist_iid(dataset_train, args.num_users) # Leads to sampling.py where data split is done
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users) # Leads to sampling.py where data split is done

  elif args.dataset == 'cifar':
        #trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trans_cifar_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trans_cifar_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar_test)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)
  elif args.dataset == 'fashion-mnist':
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True,
                                              transform=trans_fashion_mnist)
        dataset_test  = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True,
                                              transform=trans_fashion_mnist)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
  elif args.dataset == 'femnist':
        dataset_train = FEMNIST(train=True)
        dataset_test = FEMNIST(train=False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        if args.iid:
            exit('Error: femnist dataset is naturally non-iid')
        else:
            print("Warning: The femnist dataset is naturally non-iid, you do not need to specify iid or non-iid")
  elif args.dataset == 'shakespeare':
        dataset_train = ShakeSpeare(train=True)
        dataset_test = ShakeSpeare(train=False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        if args.iid:
            exit('Error: ShakeSpeare dataset is naturally non-iid')
        else:
            print("Warning: The ShakeSpeare dataset is naturally non-iid, you do not need to specify iid or non-iid")
  else:
        exit('Error: unrecognized dataset')

  """
    The img_size is typically used for:Defining Model Architectures:
    dataset_train[0][0]:
    This extracts only the image part of the tuple.
    .shape:
    This gives the dimensions of the image tensor. For example:
    For MNIST, it would return (1, 28, 28)
    The size of the input image determines the input dimensions for models such as MLPs (multi-layer perceptrons) or CNNs (convolutional neural networks).
  """
  img_size = dataset_train[0][0].shape

######################################################################################
  # SET 2: ASSIGN MODEL: This step is simply for selecting model AND assigning the global model
######################################################################################

  if args.model == 'cnn' and args.dataset == 'cifar':
      net_glob = CNNCifar(args=args).to(args.device) # Assigns the model as defined in models.py inside utils folder and send to device
  elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
      net_glob = CNNMnist(args=args).to(args.device)
  elif args.dataset == 'femnist' and args.model == 'cnn':
      net_glob = CNNFemnist(args=args).to(args.device)
  elif args.dataset == 'shakespeare' and args.model == 'lstm':
      net_glob = CharLSTM().to(args.device)
  elif args.model == 'mlp':
      len_in = 1
      for x in img_size:
          len_in *= x
      net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
  else:
      exit('Error: unrecognized model')


######################################################################################
  # SET 3: TRAIN MODEL
######################################################################################

  print(net_glob)
  net_glob.train() # Set the model to train
  
  # copy weights
  w_glob = net_glob.state_dict()
  """
  #######################################################################
  acc_test: An empty list to store testing accuracy after each global round.

  clients:
  Creates a list of LocalUpdate objects, one for each client.
  Each LocalUpdate object is initialized with:
  The client’s data (dict_users[idx]).
  Training configurations (args).
  What Happens?:
  For num_users clients, the code:
  Divides the dataset so that each client has access to its own portion.
  Sets up local training for each client.
  Example: Let’s assume:
  num_users = 3.
  dict_users = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}.
  Each client owns 3 samples.
  After this line, clients would be:
  A list of LocalUpdate objects:
  clients[0]: Handles local training for Client 0 with data indices [0, 1, 2].
  clients[1]: Handles local training for Client 1 with data indices [3, 4, 5].
  clients[2]: Handles local training for Client 2 with data indices [6, 7, 8].

  Selecting Subset of Clients
  m, clients_index_array = max(int(args.frac * args.num_users), 1), range(args.num_users)
  args.frac: Fraction of clients to be selected in each round (e.g., 0.1 means 10% of clients are selected).
  args.num_users: Total number of clients in the FL setup.
  int(args.frac * args.num_users): Computes the number of clients to select (e.g., 10 clients if frac=0.1 and num_users=100).
  max(..., 1): Ensures at least one client is selected (in case frac is very small).
  clients_index_array: Represents the indices of all clients (e.g., range(100) if num_users=100).
  Example:
  Suppose args.num_users = 10 and args.frac = 0.3.
  m = max(int(0.3 * 10), 1) = 3.
  clients_index_array = range(10) = [0, 1, 2, ..., 9].

  
  #######################################################################
  """

  acc_test = [] # An empty list to store testing accuracy after each global round.
  clients = [LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
             for idx in range(args.num_users)]
  m, clients_index_array = max(int(args.frac * args.num_users), 1), range(args.num_users)

  # Loop Over Training Rounds, args.epochs: Number of FL rounds (global training iterations).
  for iter in range(args.epochs):
      w_locals, loss_locals, weight_locols= [], [], []

      # Randomly selects m clients (without replacement) for training in this round from the list clients_index_array..
      idxs_users = np.random.choice(clients_index_array, m, replace=False)

      # Training Selected Clients, Loops through the selected clients (idxs_users) and performs local training.
      for idx in idxs_users:
          w, loss = clients[idx].train(net=copy.deepcopy(net_glob).to(args.device)) # Calls the train method of the LocalUpdate class for the client.
          w_locals.append(copy.deepcopy(w))  # Collects the updated weights from all selected clients.
          loss_locals.append(copy.deepcopy(loss)) # Collects the training losses from all selected clients.
          weight_locols.append(len(dict_users[idx])) # Collects the size of each client’s dataset (used for weighted averaging)

      # update global weights
      w_glob = FedWeightAvg(w_locals, weight_locols) # Aggregates the weights from all selected clients using weighted averaging.
      # copy weight to net_glob
      net_glob.load_state_dict(w_glob) # Updates the global model (net_glob) with the aggregated weights.

      # print accuracy
      net_glob.eval()
      acc_t, loss_t = test_img(net_glob, dataset_test, args) # This test_img function is used to evaluate the performance of the global model (net_g) on a given test dataset (datatest). It calculates the test loss and accuracy of the model and optionally prints the results if args.verbose is set.
      print("Round {:3d},Testing accuracy: {:.2f}".format(iter, acc_t))

      acc_test.append(acc_t.item())

  rootpath = './log'
  if not os.path.exists(rootpath):
      os.makedirs(rootpath)
  accfile = open(rootpath + '/accfile_fed_{}_{}_{}_iid{}.dat'.
                 format(args.dataset, args.model, args.epochs, args.iid), "w")

  for ac in acc_test:
      sac = str(ac)
      accfile.write(sac)
      accfile.write('\n')
  accfile.close()
