import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import BloodMNIST

def get_dataloaders(batch_size=32, download=True):
    """
    Loads BloodMNIST dataset (8 classes of blood cell morphology).
    Converted to 1-channel grayscale (28x28).
    """
    data_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # <-- ADD THIS LINE
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = BloodMNIST(split='train', transform=data_transform, download=download)
    val_dataset = BloodMNIST(split='val', transform=data_transform, download=download)
    test_dataset = BloodMNIST(split='test', transform=data_transform, download=download)

    class_labels = train_dataset.info['label']
    num_classes = len(class_labels)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, num_classes, class_labels