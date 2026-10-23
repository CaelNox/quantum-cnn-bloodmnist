import torch
import torch.nn as nn

class ClassicalCNN(nn.Module):
    """
    Baseline Classical 2D CNN with comparable parameter count.
    Processes raw (1, 28, 28) BloodMNIST grayscale images.
    """
    def __init__(self, num_classes=8):
        super(ClassicalCNN, self).__init__()
        self.features = nn.Sequential(
            # First Conv layer acting as classical equivalent to 2x2 stride-2 quantum filter
            nn.Conv2d(1, 4, kernel_size=2, stride=2), # Output: (4, 14, 14)
            nn.BatchNorm2d(4),
            nn.ReLU(),

            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output: (16, 7, 7)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)) # Output: (32, 4, 4)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        out = self.classifier(feat)
        return out