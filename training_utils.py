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
    data_root = 'human_poses_data'
    img_train_dir = os.path.join(data_root, 'img_train')
    categories_file = os.path.join(data_root, 'activity_categories.csv')
    train_answers_file = os.path.join(data_root, 'train_answers.csv')
    img_size = (224, 224)
    batch_size = 32
    num_workers = 8
    test_size = 0.15
    val_size = 0.15
    random_state = 42
    num_classes = 16
    lr = 0.001
    weight_decay = 1e-4
    epochs = 50
    flip_prob = 0.5
    save_dir = 'saved_models_densenet'
    growth_rate = 12
    block_config = (4, 4, 4)
    bn_size = 4
    drop_rate = 0.2


class HumanPoseDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.label_map = pd.read_csv(Config.categories_file).set_index('img_id')['category'].to_dict()

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
            print(f"Error loading {img_path}: {str(e)}")
            image = Image.new('RGB', Config.img_size)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)