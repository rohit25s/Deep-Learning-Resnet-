import os
import pickle
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from typing import Any, Callable, Optional, Tuple
from PIL import Image
from ImageUtils import preprocess_image
from Configure import preprocess_config


"""This script implements the functions for reading data.
"""

def load_data(data_dir, preprocess_config):
    """Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches
            are stored.

    Returns:
        x_train: An numpy array of shape [50000, 3072].
            (dtype=np.float32)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072].
            (dtype=np.float32)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int32)
    """

    preprocess_train, preprocess_test = preprocess_image(preprocess_config)

    train_dataset = datasets.CIFAR10(root=data_dir,
                                     train=True,
                                     transform=preprocess_train,
                                     download=True)
    
    test_dataset = datasets.CIFAR10(root=data_dir,
                                    train=False,
                                    transform=preprocess_test,
                                    download=True)
    
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=preprocess_config["batch_size"],
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=2)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=preprocess_config["batch_size"],
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=2)

    return train_loader, test_loader


def load_testing_images(data_dir, preprocess_config):
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        x_test: An numpy array of shape [N, 32, 32, 3].
            (dtype=np.float32)
    """

    ### YOUR CODE HERE
    _, preprocess_test = preprocess_image(preprocess_config)

    private_test_dataset = CIFAR10_PRIVATE(root=data_dir,
                                    transform=preprocess_test)
    
    private_test_loader = torch.utils.data.DataLoader(dataset=private_test_dataset.data,
                                              batch_size=preprocess_config["batch_size"],
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=2)

    return private_test_loader
    ### END CODE HERE


def train_valid_split(x_train, y_train, train_ratio=0.8):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        train_ratio: A float number between 0 and 1.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    
    ### YOUR CODE HERE

    ### END CODE HERE
    pass
    #return x_train_new, y_train_new, x_valid, y_valid




class CIFAR10_PRIVATE(Dataset):
    private_data = 'private_test_images_v3.npy'

    def __init__(self, root: str, transform: Optional[Callable] = None) -> None:
        private_test_path = os.path.join(root, self.private_data)
        self.data = np.load(private_test_path)
        self.transform = transform


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self) -> int:
        return len(self.data)


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict