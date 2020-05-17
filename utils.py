import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from skimage import io
import numpy as np

class PCamDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.labels = pd.read_csv(root_dir + csv_file).dropna()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        csv_name = self.labels.iloc[idx, 0]
        label = self.labels.iloc[idx, 1]

        img_name = os.path.join(
            self.root_dir,
            "all/{}.tif".format(csv_name)
        )
        image = io.imread(img_name) # 96, 96, 3
        label = torch.from_numpy(np.array([label]))
        label = label.type('torch.FloatTensor')
        
        if self.transform is not None:
            image = self.transform(image) # 3, 96, 96 with transforms.ToTensor()
        return image, label
