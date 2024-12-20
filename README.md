# MNIST Classification Project

A PyTorch implementation of MNIST digit classification using multiple model architectures. The project includes training on both local and EC2 instances.

## Project Structure

```
project/
├── models/
│ ├── init.py
│ ├── model_1.py (Baseline)
│ ├── model_2.py (Intermediate)
│ └── model_3.py (Advanced)
├── data/│ └── MNIST/
│ └── raw/│ 
├── train-images-idx3-ubyte
│ ├── train-labels-idx1-ubyte
│ ├── t10k-images-idx3-ubyte
│ └── t10k-labels-idx1-ubyte
├── datafile.py
├── train.py
└── requirements.txt
```

## Models Overview

### Model_1 (Baseline)

- Simple CNN architecture
- Parameters: 26,698
- Best Accuracy: 98.63%
- Training Time: 2 epochs
- Features:
  - 2 Convolutional layers
  - 2 Fully connected layers
  - MaxPooling
  - ReLU activation

### Model_2 (Intermediate)

- Optimized architecture with regularization
- Parameters: 15,106
- Best Accuracy: 98.80%
- Training Time: 11-12 epochs
- Features:
  - Reduced channel sizes
  - Dropout (0.2)
  - Better parameter efficiency

### Model_3 (Advanced)

- Highly optimized architecture
- Parameters: 4,506
- Target Accuracy: >99.4%
- Features:
  - Batch Normalization
  - Strategic dropout
  - Efficient channel progression

## Training Configuration

### Data Loading (datafile.py)

```
transform = transforms.Compose([transforms.ToTensor(),
transforms.Normalize((0.1307,), (0.3081,)) # MNIST mean and std])
train_loader = DataLoader(train_data,batch_size=128,shuffle=True,num_workers=2,pin_memory=True)
```

### Training Parameters

- Optimizer: Adam
- Learning Rate: Model specific (0.001 - 0.01)
- Scheduler: ReduceLROnPlateau
  - mode: 'max'
  - factor: 0.1
  - patience: 3
  - verbose: True
- Device: Automatic CUDA/CPU detection

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
Epoch: 1/15
Training Loss: 0.145
Training Accuracy: 97.85%
Validation Accuracy: 98.32%
--------------------------------------------------
```

## Model Checkpointing

- Best models are saved automatically:
  - `best_Model_3.pth`

## Features

1. **Progress Tracking**

   - TQDM progress bars
   - Real-time metrics
   - Loss and accuracy monitoring
2. **Device Management**

   - Automatic CUDA detection
   - CPU fallback
   - Proper device handling
3. **Training Optimization**

   - Learning rate scheduling
   - Model checkpointing
   - Early stopping capability

## Requirements

- PyTorch
- torchvision
- tqdm
- torchsummary

## Future Improvements

1. Add data augmentation
2. Implement cross-validation
3. Add visualization tools
4. Experiment with different optimizers
5. Add model ensemble capabilities

## EC2 Training

The project supports training on AWS EC2 instances:

- Instance setup instructions included
- SSH key management
- Proper permissions handling
