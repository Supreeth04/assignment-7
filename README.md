# MNIST Classification with PyTorch

A PyTorch implementation of MNIST digit classification focusing on efficient model architectures and training strategies.

## Project Structure

├── models/

│ ├── init.py

│ ├── model_1.py (Baseline Model)

│ └── model_2.py (Optimized Model)

├── data/

│ └── MNIST/

├── datafile.py

├── train.py

└── requirements.txt

## Models Overview

### Model_1 (Baseline)

class Model_1(nn.Module):

def init(self):

super(Model_1, self).init()

self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)

self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)

self.pool = nn.MaxPool2d(2, 2)

self.fc1 = nn.Linear(16 7 7, 32)

self.fc2 = nn.Linear(32, 10)

- Parameters: 15,106
- Best Accuracy: 98.80%
- Training Time: 11-12 epochs
- Features: Added dropout and optimized architecture

### Model_2 (Optimized)

class Model_2(nn.Module):

def init(self):

super(Model_2, self).init()

self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)

self.conv2 = nn.Conv2d(6, 12, kernel_size=3, padding=1)

self.pool = nn.MaxPool2d(2, 2)

self.dropout = nn.Dropout(0.2)

self.fc1 = nn.Linear(12 7 7, 24)

self.fc2 = nn.Linear(24, 10)

- Parameters: 15,106
- Best Accuracy: 98.80%
- Training Time: 11-12 epochs
- Features: Added dropout and optimized architecture

## Training Configuration

### Data Preprocessing

transform = transforms.Compose([

transforms.ToTensor(),

transforms.Normalize((0.1307,), (0.3081,))

])

### Training Parameters

- Batch Size: 128
- Optimizer: Adam
- Learning Rate Scheduler: ReduceLROnPlateau
  - mode: 'max'
  - factor: 0.1
  - patience: 3
  - verbose: True
- Loss Function: CrossEntropyLoss

## Requirements

torch

torchvision

tqdm

torchsummary

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train models:

```bash
python train.py
```

3. Monitor training progress:

```
Epoch: 1/12
Training Loss: 0.145
Training Accuracy: 97.85%
Validation Accuracy: 98.32%
--------------------------------------------------
```

## Model Performance

### Model_1 Results

- Quick convergence (2 epochs)
- Good baseline accuracy
- Higher parameter count
- Simple architecture

### Model_2 Results

- Better accuracy than Model_1
- Reduced parameters by ~43%
- More stable training
- Better generalization

## Features

1. **Progress Tracking**

   - Real-time training metrics
   - Validation accuracy monitoring
   - Loss tracking
   - Parameter counting
2. **Model Saving**

   - Automatic checkpointing
   - Best model preservation
   - Easy model loading
3. **Training Optimization**

   - Learning rate scheduling
   - Early stopping capability
   - Device agnostic (CPU/CUDA)

## Future Improvements

1. Add data augmentation
2. Implement cross-validation
3. Add visualization tools
4. Try different optimizers
5. Experiment with other architectures

## Contributing

Feel free to open issues and pull requests for:

- Bug fixes
- New features
- Documentation improvements
- Performance optimizations

## License

This project is open-source and available under the MIT License.
