import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm


class Config:
    data_root = os.path.join(os.path.dirname(os.getcwd()), 'human_poses_data')
    img_train_dir = os.path.join(data_root, 'img_train')
    categories_file = os.path.join(data_root, 'activity_categories.csv')
    train_answers_file = os.path.join(data_root, 'train_answers.csv')
    img_size = (224, 224)
    batch_size = 64
    num_workers = 4
    test_size = 0.15
    val_size = 0.15
    random_state = 42
    num_classes = 16
    lr = 1e-3
    weight_decay = 1e-4
    epochs = 50
    flip_prob = 0.5
    save_dir = 'saved_models_effnet_se'
    patience = 5


class HumanPoseDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row['img_id']).strip()
        label = row['target_feature']
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            image = Image.new('RGB', Config.img_size)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)