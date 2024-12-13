import torch
import torch.nn as nn
import torch.nn.functional as F

class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(12 * 7 * 7, 24)
        self.fc2 = nn.Linear(24, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 12 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

"""
TARGET:
- Achieve 98.63% accuracy
- Keep parameters under 20,000
- Complete within 12 epochs

RESULT:
- Best accuracy achieved is 98.80% in 9th epoch
- Parameters: 15106
- Epochs needed: 11

ANALYSIS:
- Added dropout for better regularization
- Reduced channel sizes to optimize parameter count
- More stable training in comparision to model_1 with adjusted learning rate (0.005)
- Model is still being little wonky and the model is underfitting.
- Model can be further trained to achieve higher accuracy.
"""
