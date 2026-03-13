from medmnist import INFO, Evaluator, ChestMNIST # type: ignore
from torchvision import transforms # type: ignore
from torch.utils.data import DataLoader # type: ignore
# import numpy as np # type: ignore  # Uncomment if using numpy arrays

# Settings - matching autoencoder.ipynb
img_size = 224               # image size to match autoencoder (28, 64, 128, or 224)
batch_size = 32              # batch size for DataLoader
download = True

# Transform - same as autoencoder
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load datasets with specified image size
train_dataset = ChestMNIST(split='train', size=img_size, transform=transform, download=download)
test_dataset = ChestMNIST(split='test', size=img_size, transform=transform, download=download)

# Only print if running as main script (not when importing)
if __name__ == "__main__":
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

# Create DataLoaders - compatible with autoencoder
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# Only print if running as main script (not when importing)
if __name__ == "__main__":
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

# Optional: If you still need numpy arrays for classical ML
# Uncomment the lines below
# X_train, y_train = [], []
# for img, label in train_dataset:
#     img_np = img.numpy()
#     img_flat = img_np.flatten()
#     X_train.append(img_flat)
#     y_train.append(label)
# X_train = np.array(X_train)
# y_train = np.array(y_train)
# print("X_train shape:", X_train.shape)
# print("y_train shape:", y_train.shape)