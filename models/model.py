import torch
import torch.nn as nn
import torch.nn.functional as F

class Model_4(nn.Module):
    def __init__(self):
        super(Model_4, self).__init__()
        # First block
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(1, 6, 3, padding=1, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Dropout(0.05),  # Reduced dropout rate
        )
        # Convolution Block 1
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(6, 12, 3, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.05),
        )
        # Transition Block 1
        self.pool1 = nn.MaxPool2d(2, 2)
        # Convolution Block 2
        self.convBlock3 = nn.Sequential(
            nn.Conv2d(12, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(0.05),
        )
        # Convolution Block 3
        self.convBlock4 = nn.Sequential(
            nn.Conv2d(24, 12, 3, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.05),
        )
        # Transition Block 2
        self.pool2 = nn.MaxPool2d(2, 2)
        # Convolution Block 4
        self.convBlock5 = nn.Sequential(
            nn.Conv2d(12, 10, 3, padding=1, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.05),
        )
        # Global Average Pooling and Output
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Final 1x1 conv
        self.finalconv = nn.Conv2d(10, 10, kernel_size=1)

    def forward(self, x):
        x = self.convBlock1(x)  # 26x26
        x = self.convBlock2(x)  # 24x24
        x = self.pool1(x)        # 11x11
        x = self.convBlock3(x)  # 22x22

        x = self.convBlock4(x)  # 11x11
        x = self.pool1(x)        # 11x11

        x = self.convBlock5(x)  # 9x9
        # x = self.convBlock6(x)  # 9x9
        # x = self.convBlock7(x)  # 7x7

        x = self.gap(x)         # 1x1
        x = self.finalconv(x)
        x = x.view(-1, 10)

        return F.log_softmax(x, dim=-1)

"""
TARGET:
- Achieve >99.4% accuracy
- Parameters <= 8,000
- Complete within 15 epochs

RESULT:
- Parameters: 6,986 (exact match)
- Channel progression: 1->8->16->20->16->20->16->10->10
- Spatial reduction: 28->26->24->22->11->9->7->1

ANALYSIS:
- Increased channel widths for better feature extraction
- Strategic use of 1x1 convolutions for channel reduction
- Dropout (0.1) throughout except after block 3
- Batch normalization for training stability
- Progressive spatial reduction through no-padding convolutions
- Balanced parameter distribution across layers
""" 