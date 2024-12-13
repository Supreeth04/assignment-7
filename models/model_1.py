import torch
import torch.nn as nn
import torch.nn.functional as F

class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 7 * 7, 32)
        self.fc2 = nn.Linear(32, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

"""
TARGET: 
- Achieve 97% accuracy as baseline
- Keep parameters under 30,000
- Complete within 10 epochs

RESULT:
- Achieved 98.63% accuracy
- Parameters: 26,698
- Epochs needed: 2

ANALYSIS:
- Model shows good initial performance but has room for improvement
- Parameter count is well within limit, allowing space for optimization
- Learning rate of 0.01 seems too aggressive, causing fluctuations
"""