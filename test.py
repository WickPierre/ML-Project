import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ========================
# CONFIGURATION
# ========================
class Config:
    # Hardware
    DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # Paths
    DATA_DIR = "human_poses_data"
    TRAIN_CSV = os.path.join(DATA_DIR, "train_answers.csv")
    CATEGORIES_CSV = os.path.join(DATA_DIR, "activity_categories.csv")
    TRAIN_IMG_DIR = os.path.join(DATA_DIR, "img_train")
    TEST_IMG_DIR = os.path.join(DATA_DIR, "img_test")

    # Model
    PRETRAINED = True
    NUM_CLASSES = 20  # По количеству категорий в файле
    DROPOUT = 0.2

    # Training
    BATCH_SIZE = 64
    EPOCHS = 30
    LR = 3e-4
    WEIGHT_DECAY = 1e-5

    # Image processing
    IMG_SIZE = 380  # Подходит для EfficientNet-B4
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]


# ========================
# DATA PREPARATION
# ========================
class ActivityDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.le = LabelEncoder()
        self.labels = self.le.fit_transform(df['target_feature'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = f"{self.df.iloc[idx]['img_id']}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def create_datasets():
    # Загрузка данных
    train_df = pd.read_csv(Config.TRAIN_CSV)
    categories = pd.read_csv(Config.CATEGORIES_CSV)

    # Фильтрация классов с малым количеством примеров
    class_counts = train_df['target_feature'].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    filtered_df = train_df[train_df['target_feature'].isin(valid_classes)]

    # Кодирование меток
    le = LabelEncoder()
    filtered_df['encoded_labels'] = le.fit_transform(filtered_df['target_feature'])

    # Разделение с учетом баланса классов
    try:
        train_data, val_data = train_test_split(
            filtered_df,
            test_size=0.15,
            stratify=filtered_df['encoded_labels'],
            random_state=42
        )
    except ValueError:
        # Fallback для проблемных классов
        train_data, val_data = train_test_split(filtered_df, test_size=0.15, random_state=42)

    # Трансформации
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(Config.MEAN, Config.STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(Config.MEAN, Config.STD)
    ])

    # Создание Dataset объектов
    train_dataset = ActivityDataset(train_data, Config.TRAIN_IMG_DIR, train_transform)
    val_dataset = ActivityDataset(val_data, Config.TRAIN_IMG_DIR, val_transform)

    return train_dataset, val_dataset, le


# ========================
# MODEL ARCHITECTURE
# ========================
def create_model():
    model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT if Config.PRETRAINED else None)

    # Модификация головы
    model.classifier = nn.Sequential(
        nn.Dropout(p=Config.DROPOUT, inplace=True),
        nn.Linear(model.classifier[1].in_features, Config.NUM_CLASSES)
    )

    return model.to(Config.DEVICE)


# ========================
# TRAINING SETUP
# ========================
def train_model(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    best_acc = 0.0

    for epoch in range(Config.EPOCHS):
        # Training
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        scheduler.step(val_acc)

        print(f'Epoch {epoch + 1}/{Config.EPOCHS}')
        print(f'Train Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f}')

        # Сохранение лучшей модели
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    print(f'Best Validation Accuracy: {best_acc:.4f}')


# ========================
# MAIN EXECUTION
# ========================
if __name__ == '__main__':
    torch.set_num_threads(Config.NUM_WORKERS)
    torch.backends.cudnn.benchmark = True

    train_dataset, val_dataset, label_encoder = create_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

    model = create_model()
    train_model(model, train_loader, val_loader)