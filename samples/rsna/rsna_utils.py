"""
Mask R-CNN
Data preparation code for RSNA Pneumonia dataset.

Copyright (c) 2020 PersonalTouch, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Phong Vu
"""

import os
import random
import pandas as pd

from sklearn.model_selection import train_test_split
from datetime import datetime

class RsnaUtils:
    """
    Utility class for the Rsna preparation.
    """
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.ann_dir = dataset_dir
    
    @staticmethod
    def hu_segmentation(image, min_hu, max_hu):
        '''
        Returns:
        image: image array containing HU values in the range [0,1]
        '''
        image = (image - min_hu) / (max_hu - min_hu)
        image[image > 1] = 1
        image[image < 0] = 0 
        return image
    
    @staticmethod
    def transform_to_hu(medical_image, image):   
        if 'RescaleIntercept' in medical_image:
            intercept = medical_image.RescaleIntercept  
        else:
            intercept = -1024
        if 'RescaleSlope' in medical_image:
            slope = medical_image.RescaleSlope
        else:
            slope = 1

        hu_image = image * slope + intercept
        return hu_image
    
    def prepare_label(self):
        """
        Prepare the training labels. Sort by Target

        Returns:
        df: Pandas dataframe to manipulate
        """
        df = pd.read_csv(os.path.join(self.ann_dir, "train_labels.csv"))
        df = df.sort_values(by='Target')

        return df
    
    def prepare_rsna(self):
        """
        Prepare the RSNA dataset for loading

        Returns:
        X_train: dataframe to build the train set (Target == 0)
        X_val: dataframe to build the val set (Target == 0)
        Y_train: dataframe to build the train set (Target == 1)
        Y_val: dataframe to build the val set (Target == 1)
        """
        # Load annotations and sort by Target column
        df = self.prepare_label()

        # Filter
        mask = df['Target'] == 0
        # Dataframe containing Target == 0
        df1 = df[mask]
        # Dataframe containing Target == 1
        df2 = df[~mask]
        # Drop columns with NaN values
        df1 = df1.dropna(1, 'all')
        df1 = df[0:6000]

        # Drop duplicate rows based on patient Id
        df2 = df2.drop_duplicates('patientId')

        print("Number of person with pneumonia:", df2.shape[0])
        print("Others:", df1.shape[0])

        # Random seed
        random.seed(datetime.now())

        # 70% luck 30% skill X == 0 Y == 1 (Target)
        X_train, X_val = train_test_split(df1, test_size=0.3, random_state=random.randint(0, 69))
        Y_train, Y_val = train_test_split(df2, test_size=0.3, random_state=random.randint(0, 69))

        return X_train, X_val, Y_train, Y_val
