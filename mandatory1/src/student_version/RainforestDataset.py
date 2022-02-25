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
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RainforestDataset(Dataset):
    def __init__(self, root_dir, trvaltest, transform):


        classes, num_classes = get_classes_list()
        
        # TODO Binarise your multi-labels from the string. HINT: There is a useful sklearn function to
        # help you binarise from strings.

        #########################
        self.root_dir  = root_dir
        self.transform = transform
        
        dataframe = pd.read_csv(root_dir + "train_v2.csv")
        img_name  = dataframe.iloc[:, 0]
        labels         = dataframe.iloc[:, 1]
        labels         = [string.split() for string in labels]

        binarizer      = preprocessing.MultiLabelBinarizer()

        self.labels    = binarizer.fit_transform(labels) 
        #########################

        # TODO Perform a test train split. It's recommended to use sklearn's train_test_split with the following
        # parameters: test_size=0.33 and random_state=0 - since these were the parameters used
        # when calculating the image statistics you are using for data normalisation.
        
        #for debugging you can use a test_size=0.66 - this trains then faster
        

        # OR optionally you could do the test train split of your filenames and labels once, save them, and
        # from then onwards just load them from file.


        #########################

        data_train, data_val, label_train, label_val = train_test_split(img_name, self.labels, test_size = 0.33, random_state = 0)
        if trvaltest == 0:      # Training mode
            self.img_filenames = list(data_train)
            self.labels   = list(label_train)
        else:                   # Validation and "test" (since we sloppily mix these to according to the exerciese text) mode
            self.img_filenames = list(data_val)
            self.labels   = list(label_val)

        #########################


    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        # TODO get the label and filename, and load the image from file.

        #########################

        labels = self.labels[idx]
        
        with PIL.Image.open(self.root_dir + "train-tif-v2/" + self.img_filenames[idx] + ".tif") as img:
            #img = np.asarray(img).astype(np.uint8)
        
            
            #print(np.asfarray(img, dtype = np.uint8).shape)
            #print(np.asfarray(img).astype(np.uint8).dtype)


            if self.transform:
                img = self.transform(img)
            #########################

            sample = {'image': img,
                    'label': labels,
                    'filename': self.img_filenames[idx]}
        return sample
