"""
This code defines functions for sampling data into IID (Independent and Identically Distributed) 
or Non-IID (Non-Independent and Identically Distributed) formats.

IID:
Data is randomly distributed across all users without considering labels or specific patterns. 
Each user receives a dataset that is statistically similar to others.

Non-IID:
Data is distributed in a way that creates label or feature imbalances among users. 
For example, one user might only have images of specific digits in MNIST.

Shards:
In Non-IID sampling, the dataset is divided into smaller groups (shards) based on labels. 
Each user is assigned a subset of these shards, creating a non-uniform distribution.
"""

import numpy as np
from torchvision import datasets, transforms

    """
    ##############################################
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    ##############################################
    """
  
def mnist_iid(dataset, num_users):
    
    dict_users = {} 
    """
    ###############################################
      This initializes an empty dictionary dict_users that will eventually map user IDs (keys) to their respective data indices (values).
        dict_users = {
          0: {3, 18, 56, ...},  # Data indices assigned to user 0
          1: {4, 22, 67, ...},  # Data indices assigned to user 1
          ...
      }
    ##############################################
    """
  
    num_items = int(len(dataset) / num_users)
  
    """
    #############################################
      This determines how many data samples each user will receive.
      len(dataset) gives the total number of samples in the dataset.
      num_users specifies the total number of users to which the data will be distributed.
      Dividing len(dataset) by num_users gives the number of samples per user (num_items).
      If the dataset has 60,000 samples (e.g., MNIST) and there are 100 users:
      num_items = int(60000 / 100)
      num_items = 600  # Each user gets 600 samples
    #############################################
    """

    all_idxs = [i for i in range(len(dataset))]
  
    """
    #############################################
      This creates a list of all sample indices in the dataset.
      range(len(dataset)) generates numbers from 0 to len(dataset) - 1, which correspond to the indices of all data samples in the dataset.
      The list comprehension [i for i in range(len(dataset))] converts this range object into a list.
      This list will be used to randomly assign data indices to users.
      For a dataset with 10 samples:
      len(dataset) = 60000
      all_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, ..., 59999]  # List of indices
    #############################################
    """
  
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

    """
     #############################################
      Randomly selects a subset of indices from all_idxs and assigns it to user i
      np.random.choice(all_idxs, num_items, replace=False):
      Randomly selects num_items indices from the list all_idxs.
      replace=False ensures no index is selected more than once during this sampling process.
      Example:
      If all_idxs = [0, 1, 2, ..., 9] and num_items = 3:
      np.random.choice(all_idxs, num_items, replace=False)
      # Output: [3, 7, 1]  # Example of randomly selected indices
      set():
      Converts the list of selected indices into a set. Using a set ensures fast lookups and efficient operations when removing indices later.
      dict_users[i] = ...:
      Assigns the set of randomly chosen indices to user i in the dictionary dict_users.
      Example:
      dict_users[0] = {3, 7, 1}  # Indices assigned
      set(all_idxs) - dict_users[i]: Performs a set difference operation, removing all indices in dict_users[i] from the available indices.
      Example:
      If all_idxs = [0, 1, 2, ..., 9] and dict_users[0] = {3, 7, 1}:
      set(all_idxs) - dict_users[0]
      # Output: {0, 2, 4, 5, 6, 8, 9} 
      list(...): Converts the resulting set of indices back into a list.
      all_idxs = ...: Updates all_idxs with the remaining indices.
      Example:
      all_idxs = [0, 2, 4, 5, 6, 8, 9]
     #############################################
    """

def mnist_noniid(dataset, num_users):
    """
    #############################################
    num_shards: The total number of "shards" (small subsets) to divide the dataset into.
    num_imgs: The number of images per shard.
    Example: If num_users = 10, then num_shards = 20.
    num_imgs: The number of images per shard.
    Example: If the dataset contains 60,000 images, and there are 20 shards, each shard will contain 60,000 / 20 = 3,000 images.
    idx_shard: This creates a list of integers from 0 to num_shards - 1.
    If num_shards = 20, then:
    idx_shard = [0, 1, 2, ..., 19]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}: Creates keys for each user (from 0 to num_users - 1).This ensures that:
    Later in the code, indices can be added to this array for each user.
    The NumPy array is explicitly defined as type int64, which helps avoid type issues when concatenating data later.
    For example, after this step:
    dict_users = {
    0: np.array([], dtype='int64'),  # User 0's data
    1: np.array([], dtype='int64'),  # User 1's data
    ...
    num_users-1: np.array([], dtype='int64'),  # Last user's data
    }
    Federated learning requires dividing the dataset into multiple shards (groups) to simulate the behavior of different users owning non-IID data.
    idxs helps organize these data samples into shards and assign them to users.
    idxs = np.arange(num_shards * num_imgs): Creates a sequential array of integers from 0 to num_shards * num_imgs - 1.
    For example:
    If there are 10,000 images in the dataset, num_shards * num_imgs would also be 10,000.
    np.arange(10_000) will generate:
    [0, 1, 2, 3, ..., 9999]
    Each number corresponds to one image in the dataset. These numbers will help us split the dataset into parts later.
    labels = dataset.train_labels.numpy(): Extracts the labels (ground-truth classes) of the dataset as a NumPy array.
    #############################################
    """
   
    num_shards, num_imgs = num_users * 2, int(len(dataset) / (num_users * 2))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    """
    #############################################
    In non-IID settings, the goal is to split the dataset based on the labels to create shards with biased distributions.
    For example:
    If you want one user to primarily work with data from class 0 and another user to work with class 1, you first need to sort the dataset by labels.
    Sorting ensures that samples with the same label are grouped together, making it easier to create label-biased shards.
    
    np.vstack((idxs, labels)): Combine indices and labels
    idxs (sample indices) and labels (class labels) are stacked into a 2D array where:
    Row 0: idxs (indices of samples)
    Row 1: labels (corresponding labels of the samples)
    Suppose we have a dataset with 10 images and their labels:
    Images:  [Img0, Img1, Img2, Img3, Img4, Img5, Img6, Img7, Img8, Img9]
    Labels:  [5,    0,    4,    1,    9,    2,    1,    3,    1,    4]

    idxs = np.arange(len(dataset)) gives:
    idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Step 2: Retrieve labels
    
    labels = dataset.train_labels.numpy() gives:
    labels = [5, 0, 4, 1, 9, 2, 1, 3, 1, 4]
    Step 3: Combine idxs and labels for sorting
    
    Combine idxs and labels into a 2D array:
    idxs_labels = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],   # Indices
                   [5, 0, 4, 1, 9, 2, 1, 3, 1, 4]]  # Labels
    idxs_labels[1, :]: This extracts the second row (labels)
    .argsort(): This sorts the labels in ascending order and returns the indices of the sorted order
    sorted_indices = idxs_labels[1, :].argsort() now yields:  = [1, 3, 6, 8, 7, 5, 2, 9, 0, 4]
    idxs_labels = idxs_labels[:, sorted_indices]: This reorders all rows of idxs_labels based on the sorted order of the second row (labels)
    This gives:
    idxs_labels = [[1, 3, 6, 8, 7, 5, 2, 9, 0, 4],   # Sorted indices
                   [0, 1, 1, 1, 3, 2, 4, 4, 5, 9]]  # Sorted labels

     idxs = idxs_labels[0, :]: After sorting, the first row (indices) is extracted
    #############################################
    """
    
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    """
    #############################################
    This loop iterates through the users, where each user i is assigned a subset of data from the shards.
    rand_set = set(np.random.choice(idx_shard, 2, replace=False))
    Randomly selects 2 shard indices from idx_shard (without replacement) to assign to the current user.
    Example: If idx_shard = [0, 1, 2, 3, 4], this might select {1, 3}.
    idx_shard = list(set(idx_shard) - rand_set): Remove Selected Shards from idx_shard. Updates idx_shard to remove the shards already assigned to the current user, ensuring no shard is assigned to more than one user.
    
    For each shard in rand_set:
    Calculate the range of indices corresponding to that shard:
    idxs[rand * num_imgs:(rand + 1) * num_imgs]
    Example: If rand = 1 and num_imgs = 200, this selects idxs[200:400] (the 200 images in shard 1).
    Add these indices to dict_users[i] for the current user using np.concatenate.
    Assume:
    num_users = 3
    num_shards = 6
    num_imgs = 200
    idxs = [0, 1, ..., 1199] (sorted indices)
    Iteration 1
    
    User 0:
    Randomly selects rand_set = {1, 4}.
    Assigns idxs[200:400] (shard 1) and idxs[800:1000] (shard 4) to User 0.
    dict_users[0] = [200, 201, ..., 399, 800, 801, ..., 999].
    Iteration 2
    
    User 1:
    Randomly selects rand_set = {0, 5}.
    Assigns idxs[0:200] (shard 0) and idxs[1000:1200] (shard 5) to User 1.
    dict_users[1] = [0, 1, ..., 199, 1000, 1001, ..., 1199].
    Iteration 3
    
    User 2:
    Remaining shards are {2, 3}.
    Assigns idxs[400:600] (shard 2) and idxs[600:800] (shard 3) to User 2.
    dict_users[2] = [400, 401, ..., 799].
    Final dict_users
    
    {
      0: [200, 201, ..., 399, 800, 801, ..., 999],
      1: [0, 1, ..., 199, 1000, 1001, ..., 1199],
      2: [400, 401, ..., 799]
    }
    #############################################
    """
  
    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users

def fashion_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dict_users = {}
    num_items = int(len(dataset) / num_users)
    all_idxs = [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def fashion_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = num_users * 2, int(len(dataset) / (num_users * 2))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dict_users = {}
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = num_users * 2, int(len(dataset) / (num_users * 2))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users

