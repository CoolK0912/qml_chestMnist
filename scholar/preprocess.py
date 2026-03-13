"""
Data preprocessing for chest X-ray classification
This file should be customized based on your actual data location and format
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# ============================================================================
# CONFIGURATION - MODIFY THESE PATHS FOR YOUR DATA
# ============================================================================

# Path to your dataset on Scholar cluster
DATA_ROOT = "/path/to/your/data"  # CHANGE THIS!
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

# If using a specific dataset like ChestX-ray14 or CheXpert
# Uncomment and modify as needed:
# TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
# TEST_CSV = os.path.join(DATA_ROOT, "test.csv")

IMG_SIZE = 224
N_CLASSES = 14

# ============================================================================
# DATA TRANSFORMS
# ============================================================================

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])  # Grayscale normalization
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# ============================================================================
# DATASET CLASS
# ============================================================================

class ChestXrayDataset(Dataset):
    """
    Custom dataset for chest X-ray images.
    Modify this based on your actual data structure.
    """
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        
        # Load your data here
        # This is a placeholder - modify based on your actual data format
        self.images = []
        self.labels = []
        
        # Example: If your data is organized in folders by class
        # for class_idx in range(N_CLASSES):
        #     class_dir = os.path.join(root_dir, f"class_{class_idx}")
        #     if os.path.exists(class_dir):
        #         for img_name in os.listdir(class_dir):
        #             img_path = os.path.join(class_dir, img_name)
        #             self.images.append(img_path)
        #             label = np.zeros(N_CLASSES)
        #             label[class_idx] = 1
        #             self.labels.append(label)
        
        # OR if you have a CSV file with image paths and labels:
        # import pandas as pd
        # df = pd.read_csv(csv_path)
        # self.images = df['image_path'].tolist()
        # self.labels = df[['class_0', 'class_1', ..., 'class_13']].values
        
        print(f"Loaded {len(self.images)} images from {root_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
        
        # Load label
        label = self.labels[idx]
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label).float()
        else:
            label = torch.tensor(label, dtype=torch.float)
        
        return image, label


# ============================================================================
# ALTERNATIVE: Using MedMNIST (if you want to test with a public dataset)
# ============================================================================

try:
    import medmnist
    from medmnist import INFO
    
    def load_medmnist_data():
        """
        Load ChestMNIST or PathMNIST from MedMNIST as an example
        """
        # ChestMNIST has 14 classes
        data_flag = 'chestmnist'
        download = True
        
        info = INFO[data_flag]
        n_channels = info['n_channels']
        n_classes = len(info['label'])
        
        DataClass = getattr(medmnist, info['python_class'])
        
        # Load datasets
        train_dataset = DataClass(
            split='train',
            transform=train_transform,
            download=download,
            size=IMG_SIZE
        )
        
        test_dataset = DataClass(
            split='test',
            transform=test_transform,
            download=download,
            size=IMG_SIZE
        )
        
        print(f"Loaded MedMNIST dataset: {data_flag}")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
        print(f"  Number of classes: {n_classes}")
        
        return train_dataset, test_dataset
    
    # Try to use MedMNIST as fallback
    print("Attempting to load MedMNIST dataset...")
    train_dataset, test_dataset = load_medmnist_data()
    
except ImportError:
    print("MedMNIST not available. Using custom dataset loader.")
    print("Please ensure your data is properly configured in preprocess.py")
    
    # Fall back to custom dataset
    train_dataset = ChestXrayDataset(
        root_dir=TRAIN_DIR,
        transform=train_transform,
        is_train=True
    )
    
    test_dataset = ChestXrayDataset(
        root_dir=TEST_DIR,
        transform=test_transform,
        is_train=False
    )

# ============================================================================
# VERIFY DATA
# ============================================================================

if __name__ == "__main__":
    print("\nDataset Verification:")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Check a sample
    img, label = train_dataset[0]
    print(f"\nSample data:")
    print(f"  Image shape: {img.shape}")
    print(f"  Label shape: {label.shape}")
    print(f"  Label values: {label}")
    
    # Check for multi-label samples
    multi_label_count = 0
    for i in range(min(100, len(train_dataset))):
        _, label = train_dataset[i]
        if label.sum() > 1:
            multi_label_count += 1
    
    print(f"\nMulti-label samples in first 100: {multi_label_count}")
