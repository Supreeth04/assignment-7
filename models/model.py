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
        x = self.convBlock1(x)  
        x = self.convBlock2(x)  
        x = self.pool1(x)        
        x = self.convBlock3(x) 

        x = self.convBlock4(x)  
        x = self.pool1(x)       

        x = self.convBlock5(x) 
        # x = self.convBlock6(x) 
        # x = self.convBlock7(x) 

        x = self.gap(x)
        x = self.finalconv(x)
        x = x.view(-1, 10)

        return F.log_softmax(x, dim=-1)

"""
TARGET:
- Achieve >99.4% accuracy
- Parameters <= 8,000
- Complete within 15 epochs

RESULT:
- Parameters: 7,204 
- Obtained accuracy is varying between 99.1% and 99.3% every time when training is run.

ANALYSIS:
- Changed Dropout (0.05) 
- Batch normalization for training stability
- Increased learning rate to 0.05, using ReduceLROnPlateau scheduler with patience 1, factor 0.1.
- Model is underfitting, can be further optimised to achieve higher accuracy.
""" 