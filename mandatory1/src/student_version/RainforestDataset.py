from pickletools import uint8
import torch
from torch.utils.data import Dataset
import os
import PIL.Image
import numpy as np
import pandas as pd
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
import os

def get_classes_list():
    classes = ['clear', 'cloudy', 'haze', 'partly_cloudy',
               'agriculture', 'artisinal_mine', 'bare_ground', 'blooming',
               'blow_down', 'conventional_mine', 'cultivation', 'habitation',
               'primary', 'road', 'selective_logging', 'slash_burn', 'water']
    return classes, len(classes)


class ChannelSelect(torch.nn.Module):
    """This class is to be used in transforms.Compose when you want to use selected channels. e.g only RGB.
    It works only for a tensor, not PIL object.
    Args:
        channels (list or int): The channels you want to select from the original image (4-channel).

    Returns: img
    """
    def __init__(self, channels=[0, 1, 2]):
        super().__init__()
        self.channels = channels

    def forward(self, img):
        """
        Args:
            img (Tensor): Image
        Returns:
            Tensor: Selected channels from the image.
        """
        return img[self.channels, ...]

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.channels)


class RainforestDataset(Dataset):
    """A custom data loader for the Planet dataset.
    """
    def __init__(self, root_dir, trvaltest, transform):
        """Initiating data loader.
        Parameters
        ----------
        root_dir : str
            The path to the data root directory.
        trvaltest : int
            Takes value 0 or 1, for respectively loading training or testing data.
        transform : nn.Compose
            Transform to apply to images that are loaded.
        """

        classes, num_classes = get_classes_list()   # Get class names and number of classes
        
        self.root_dir  = root_dir
        self.transform = transform
        
        dataframe = pd.read_csv(root_dir + "train_v2.csv")  # opening dataframe with labels and image filenames
        img_name  = dataframe.iloc[:, 0]   # image filenames

        labels         = dataframe.iloc[:, 1]     # Extracting labels from data frame.
        labels         = [string.split() for string in labels]

        binarizer      = preprocessing.MultiLabelBinarizer()    # Transforming labels from strings to binaries, 
        self.labels    = binarizer.fit_transform(labels)        # i.e. whether a given class is present or not.

        # Splitting data into training and validation dataset.
        data_train, data_val, label_train, label_val = train_test_split(img_name, self.labels, test_size = 0.33, random_state = 0)
        
        if trvaltest == 0:      # Training mode
            self.img_filenames = list(data_train)
            self.labels   = np.array(label_train).astype(np.float32)
        else:                   # Validation and "test" (since we sloppily mix these to according to the exerciese text) mode
            self.img_filenames = list(data_val)
            self.labels   = np.array(label_val).astype(np.float32)
        

    def __len__(self):
        """Returning length of dataset

        Returns
        -------
        int
            Length of dataset.
        """
        return len(self.img_filenames)

    def __getitem__(self, idx):
        """Function that opens and loads images from 
           the Planet dataset, subsequently applying a transform to it.

        Parameters
        ----------
        idx : int
            Index of image to load

        Returns
        -------
        dict
            Dictionary containing image tensor, image labels (binarized) and image filename.
        """

        labels = self.labels[idx]   
        
        with PIL.Image.open(self.root_dir + "train-tif-v2/" + self.img_filenames[idx] + ".tif") as img:
            # Opening image and apply transform to it.
            if self.transform:
                img = self.transform(img)

            sample = {'image': img,
                    'label': labels,
                    'filename': self.img_filenames[idx]}
        return sample
