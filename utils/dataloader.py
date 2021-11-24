import os
import torch
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from .imageprocess import image_transformer, tta_transformer, image_processor
import IPython


class CustomDataLoader():
    
    def __init__(self, df, train=True, row_index=None, device=None, tta=False, angle=None):
        """
        Arguments:
        img_dir  : Where train dataset is located, e.g. /loc/of/your/path/trainset
        label_dir: Where label csv file is located, e.g. /loc/of/your/path/LABEL.csv
        train    : Used to specify trainset. For validation and testset, set this value as False.
        row_idx  : Mainly used for train/val split. Among all images it specifies which row should be processed.
            Usage INPUT:np.array()
                  e.g. np.array([1, 5, 10, 102, ...])
        use_cuda : Decide whether to use cuda. If cuda is not available, it will be set False.
        """
        # assert os.path.exists(img_dir) and os.path.exists(label_dir), "Path not exists."
        
        label_df = df
        #IPython.embed(); exit()
        self.label_index = label_df.index[row_index]
        self.label_values = label_df.iloc[row_index, 1:-1].values
        self.img_dir = df['file_name'][self.label_index]
        
        # Transformation
        self.train = train

        # Set device
        self.device = device
        
        # TTA(inference)
        self.tta = tta
        self.angle = angle
        

    def __len__(self):
        return self.label_index.__len__()
    
    
    def __getitem__(self, item_index):
        """
        This method returns [label, image] given an index.
        Returned values are of torch.tensor type.
        """
        idx = item_index

        label = self.label_values[idx]
        # IPython.embed();exit(1);
        
        image_path = self.img_dir.iloc[idx]
        assert os.path.exists(image_path), f"Given image path not exists: {image_path}"
        
        pil_image = Image.open(image_path).convert('RGB')
        
        if not self.tta:
            fin_image = image_transformer(pil_image, self.train)
        else:
            fin_image = tta_transformer(pil_image, self.angle)

        # To torch.tensor type
        image_item = fin_image.float().to(self.device)
        label_item = torch.tensor(label).float().to(self.device)
        
        return image_item, label_item
