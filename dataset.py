import torch  # PyTorch library for deep learning
from torch.utils.data import Dataset  # PyTorch dataset utilities
import numpy as np  # NumPy for numerical computations
import os  # OS module for file operations
import random  # Random module for shuffling data

class Normal_Loader(Dataset):
    """
    PyTorch Dataset class for loading normal video sequences.
    
    Args:
        is_train (int): 1 for training data, 0 for test data.
        path (str): Path to the dataset directory.
    """
    def __init__(self, is_train=1, path='/workspace/DATA/UCF-Crime/'):
        super(Normal_Loader, self).__init__()
        self.is_train = is_train
        self.path = path

        # Load the appropriate file list based on train or test mode
        if self.is_train == 1:
            data_list = os.path.join(path, 'train_normal.txt')
        else:
            data_list = os.path.join(path, 'test_normalv2.txt')
        
        with open(data_list, 'r') as f:
            self.data_list = f.readlines()
        
        # Shuffle and remove last 10 samples in test mode
        if self.is_train == 0:
            random.shuffle(self.data_list)
            self.data_list = self.data_list[:-10]
    
    def __len__(self):
        """Returns the number of samples."""
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Loads and returns a sample from the dataset at the given index.
        
        Returns:
            - If train mode: concatenated RGB and optical flow numpy arrays.
            - If test mode: concatenated RGB and optical flow numpy arrays, ground truth labels, and frame count.
        """
        if self.is_train == 1:
            rgb_npy = np.load(os.path.join(self.path + 'all_rgbs', self.data_list[idx].strip() + '.npy'))
            flow_npy = np.load(os.path.join(self.path + 'all_flows', self.data_list[idx].strip() + '.npy'))
            concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            return concat_npy
        else:
            name, frames, gts = self.data_list[idx].split(' ')[0], int(self.data_list[idx].split(' ')[1]), int(self.data_list[idx].split(' ')[2].strip())
            rgb_npy = np.load(os.path.join(self.path + 'all_rgbs', name + '.npy'))
            flow_npy = np.load(os.path.join(self.path + 'all_flows', name + '.npy'))
            concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            return concat_npy, gts, frames

class Anomaly_Loader(Dataset):
    """
    PyTorch Dataset class for loading anomalous video sequences.
    
    Args:
        is_train (int): 1 for training data, 0 for test data.
        path (str): Path to the dataset directory.
    """
    def __init__(self, is_train=1, path='/workspace/DATA/UCF-Crime/'):
        super(Anomaly_Loader, self).__init__()
        self.is_train = is_train
        self.path = path

        # Load the appropriate file list based on train or test mode
        if self.is_train == 1:
            data_list = os.path.join(path, 'train_anomaly.txt')
        else:
            data_list = os.path.join(path, 'test_anomalyv2.txt')
        
        with open(data_list, 'r') as f:
            self.data_list = f.readlines()
    
    def __len__(self):
        """Returns the number of samples."""
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Loads and returns a sample from the dataset at the given index.
        
        Returns:
            - If train mode: concatenated RGB and optical flow numpy arrays.
            - If test mode: concatenated RGB and optical flow numpy arrays, ground truth labels (list), and frame count.
        """
        if self.is_train == 1:
            rgb_npy = np.load(os.path.join(self.path + 'all_rgbs', self.data_list[idx].strip() + '.npy'))
            flow_npy = np.load(os.path.join(self.path + 'all_flows', self.data_list[idx].strip() + '.npy'))
            concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            return concat_npy
        else:
            name, frames, gts = self.data_list[idx].split('|')[0], int(self.data_list[idx].split('|')[1]), self.data_list[idx].split('|')[2][1:-2].split(',')
            gts = [int(i) for i in gts]  # Convert ground truth labels to integer list
            rgb_npy = np.load(os.path.join(self.path + 'all_rgbs', name + '.npy'))
            flow_npy = np.load(os.path.join(self.path + 'all_flows', name + '.npy'))
            concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            return concat_npy, gts, frames

if __name__ == '__main__':
    # Test Normal_Loader with test mode
    loader2 = Normal_Loader(is_train=0)
    print(len(loader2))  # Print number of samples in dataset
    # Uncomment to print first sample
    # print(loader2[1])
