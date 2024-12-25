
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
  
  clients: A list of LocalUpdate objects, each representing a client.
 
  LocalUpdate: Handles local training for each client using a subset of the dataset (dataset_train) 
  specified by the user indices (dict_users[idx]).
  The for idx in range(args.num_users) loop ensures that a LocalUpdate object is created for each user (client).
  #######################################################################
  """

  acc_test = [] # An empty list to store testing accuracy after each global round.
  clients = [LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
             for idx in range(args.num_users)]
  m, clients_index_array = max(int(args.frac * args.num_users), 1), range(args.num_users)
  for iter in range(args.epochs):
      w_locals, loss_locals, weight_locols= [], [], []
      idxs_users = np.random.choice(clients_index_array, m, replace=False)
      for idx in idxs_users:
          w, loss = clients[idx].train(net=copy.deepcopy(net_glob).to(args.device))
          w_locals.append(copy.deepcopy(w))
          loss_locals.append(copy.deepcopy(loss))
          weight_locols.append(len(dict_users[idx]))

      # update global weights
      w_glob = FedWeightAvg(w_locals, weight_locols)
      # copy weight to net_glob
      net_glob.load_state_dict(w_glob)

      # print accuracy
      net_glob.eval()
      acc_t, loss_t = test_img(net_glob, dataset_test, args)
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
