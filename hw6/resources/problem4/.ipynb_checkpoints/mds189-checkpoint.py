import torch
from torch.utils import data
import pandas as pd
import random
import json
import numpy as np
from skimage import io, transform
from PIL import Image

class Mds189(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, label_file, loader, transform):
        'Initialization'
        self.label_file = label_file
        self.loader = loader
        self.transform = transform
        self.label_map = ['reach','squat','pushup','inline',
                          'hamstrings','lunge','deadbug','stretch']
        self.data= pd.read_csv(self.label_file,header=None)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def map_label_to_int(self,y):
        'The labels need to be integers'
        label_map = {'reach_both': 0,        # the key frames are labeled with the side
                     'squat_both': 1,
                     'inline_left': 2,
                     'inline_right': 2,
                     'lunge_left': 3,
                     'lunge_right': 3,
                     'hamstrings_left': 4,
                     'hamstrings_right': 4,
                     'stretch_left': 5,
                     'stretch_right': 5,
                     'deadbug_left': 6,
                     'deadbug_right': 6,
                     'pushup_both': 7,
                     'reach': 0,            # the video frames don't have information about which side is moving 
                     'squat': 1,
                     'inline': 2,
                     'lunge': 3,
                     'hamstrings': 4,
                     'stretch': 5,
                     'deadbug': 6,
                     'pushup': 7,
                     'label': -1           # label is the placeholder in `videoframe_data_test.txt` for the kaggle frame labels
                    }
        return label_map[y]

    def __getitem__(self,idx):
        'Generates one sample of data'
        path,target = self.data.iloc[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        movement = self.map_label_to_int(target)

        return sample,movement
