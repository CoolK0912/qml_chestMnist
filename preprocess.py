from medmnist import INFO, Evaluator, ChestMNIST
from torchvision import transforms
from torch.utils.data import DataLoader

# Settings
img_size = 224               # image size to match autoencoder (28, 64, 128, or 224)
batch_size = 32              # batch size for DataLoader
download = True

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load datasets with specified image size
train_dataset = ChestMNIST(split='train', size=img_size, transform=transform, download=download)
test_dataset = ChestMNIST(split='test', size=img_size, transform=transform, download=download)
