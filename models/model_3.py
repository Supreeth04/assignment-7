import torch
import torch.nn as nn
import torch.nn.functional as F

class Model_3(nn.Module):
    def __init__(self):
        super(Model_3, self).__init__()
        # First conv block
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.dropout1 = nn.Dropout(0.15)

        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.dropout2 = nn.Dropout(0.15)
        
        # Second conv block
        self.conv3= nn.Conv2d(8, 12, kernel_size=3, padding=1)
        self.bn3= nn.BatchNorm2d(12)
        self.dropout3= nn.Dropout(0.15)

        self.conv4= nn.Conv2d(12, 16, kernel_size=3, padding=1)
        self.bn4= nn.BatchNorm2d(16)
        self.dropout4= nn.Dropout(0.15)
        
        # Final conv block (no dropout)
        self.conv5 = nn.Conv2d(16, 10, kernel_size=3, padding=1)
        self.bn5= nn.BatchNorm2d(10)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        # First block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)

        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        
        # Second block
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)

        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout4(x)
        
        # Final block (no dropout)
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Global Average Pooling
        x = self.gap(x)
        x = x.view(-1, 10)
        
        return x

"""
TARGET:
- Achieve 99.4% accuracy
- Keep parameters under 8,000
- Complete within 15 epochs

RESULT:
- Achieved accuracy >=97.90% consistently in last 3 epochs.
- Parameters: 4506
- Epochs needed: > 15

ANALYSIS:
- Replaced FC layers with Global Average Pooling for better generalization
- Added dropout after first two conv blocks but not the last
- Increased channel depth progressively (8->16->10)
- Maintained batch normalization for training stability
- Learning rate of 0.001 with ReduceLROnPlateau scheduler
- Model is underfitting, can be further optimised to achieve higher accuracy.
- Model can have increased parameters to achieve higher accuracy.
""" 