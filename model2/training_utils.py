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
    # Поменять пути, если что-то не так с загрузкой данных
    data_root = 'human_poses_data'
    img_train_dir = os.path.join(data_root, 'img_train')
    categories_file = os.path.join(data_root, 'activity_categories.csv')
    train_answers_file = os.path.join(data_root, 'train_answers.csv')
    img_size = (224, 224)
    batch_size = 32
    num_workers = 6
    test_size = 0.15
    val_size = 0.15
    random_state = 42
    num_classes = 20
    lr = 0.001
    weight_decay = 1e-4
    epochs = 50
    flip_prob = 0.5
    save_dir = 'saved_models_efficientnet2'
    growth_rate = 12
    block_config = (4, 4, 4)
    bn_size = 4
    drop_rate = 0.2  # Можно попробовать и больше, до 0.4-0.5


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


def get_transforms(config, train=True):
    base_transforms = [
        transforms.Resize(config.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    if train:
        augmentations = [
            transforms.RandomHorizontalFlip(config.flip_prob),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ]
        return transforms.Compose(augmentations + base_transforms)
    return transforms.Compose(base_transforms)


def load_data(config):
    print("\n[1/4] Загрузка данных...")
    train_df = pd.read_csv(config.train_answers_file)
    print(f"Пример данных:\n{train_df.head()}")

    print("\n[2/4] Валидация данных...")
    train_df = train_df.dropna()
    print(f"Данных после очистки: {len(train_df)}")

    print("\n[3/4] Проверка изображений...")
    existing_images = {f.split('.')[0] for f in os.listdir(config.img_train_dir)}
    print(f"Найдено изображений: {len(existing_images)}")

    train_df = train_df[train_df['img_id'].astype(str).isin(existing_images)]
    print(f"Данных после фильтрации: {len(train_df)}")

    print("\n[4/4] Разделение данных...")
    train_df, test_val_df = train_test_split(
        train_df,
        test_size=config.test_size + config.val_size,
        stratify=train_df['target_feature'],
        random_state=config.random_state
    )
    val_df, test_df = train_test_split(
        test_val_df,
        test_size=config.test_size / (config.test_size + config.val_size),
        stratify=test_val_df['target_feature'],
        random_state=config.random_state
    )

    return (
        HumanPoseDataset(train_df, config.img_train_dir, get_transforms(config, True)),
        HumanPoseDataset(val_df, config.img_train_dir, get_transforms(config, False)),
        HumanPoseDataset(test_df, config.img_train_dir, get_transforms(config, False))
    )