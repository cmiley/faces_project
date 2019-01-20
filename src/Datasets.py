from torch.utils.data import Dataset
import pandas as pd
from skimage import io
import os
import torch
from PIL import Image
import numpy as np


class CelebAttributesDataset(Dataset):
    def __init__(self, csv_file, root_dir, tf=None):
        self.attrib_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = tf

    def __len__(self):
        return len(self.attrib_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.attrib_frame.iloc[idx, 0])
        image = Image.open(img_name)

        image = self.transform(image)

        attribs = self.attrib_frame.iloc[idx, 1:].values
        attribs = (attribs + 1)/2
        attribs = attribs.astype(float)
        sample = {'image': image, 'attribs': attribs}

        # if self.transform:
        #     sample = self.transform(sample)

        return sample


class CelebLandmarksDataset(Dataset):
    def __init__(self, csv_file, root_dir, tf=None):
        self.landmark_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = tf

    def __len__(self):
        return len(self.landmark_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmark_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmark_frame.iloc[idx, 1:].values
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


class CelebBoundingDataset(Dataset):
    def __init__(self, csv_file, root_dir, tf=None):
        self.bbox_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = tf

    def __len__(self):
        return len(self.bbox_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.bbox_frame.iloc[idx, 0])
        image = io.imread(img_name)
        origin = self.bbox_frame.iloc[idx, 1:3].values.astype('float')
        width = self.bbox_frame.iloc[idx, 3].astype('float')
        height = self.bbox_frame.iloc[idx, 4].astype('float')
        opposite = [origin[0] + width, origin[1] + height]
        points = [origin, opposite]
        sample = {'image': image, 'origin': origin, 'width': width, 'height': height, 'opposite': opposite, 'points': points}

        if self.transform:
            sample = self.transform(sample)

        return sample
