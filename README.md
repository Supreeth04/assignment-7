# MNIST Classification Project

A PyTorch implementation of MNIST digit classification using multiple model architectures, with a target of achieving >99.4% accuracy using less than 8,000 parameters.

## Project Structure

├── models/

│ ├── init.py

│ ├── model_1.py (Baseline Model)

│ ├── model_2.py (Optimized Model)

│ └── model_3.py (Final Model)

├── data/

│ └── MNIST/

│ └── raw/

├── datafile.py

├── train.py

├── requirements.txt

└── .gitignore

## Models Overview

### Model_1 (Baseline)

- Simple CNN architecture
- Parameters: 26,698
- Best Accuracy: 98.63%
- Features: Basic CNN with 2 conv layers and 2 FC layers

## Training Configuration

### Data Loading (datafile.py)

transform = transforms.Compose([

transforms.ToTensor(),

transforms.Normalize((0.1307,), (0.3081,))

])

### Training Parameters

- Batch Size: 128
- Optimizer: Adam
- Learning Rate Scheduler: ReduceLROnPlateau
- Loss Function: CrossEntropyLoss
- Device: CUDA/CPU auto-detection

## Requirements

torch

torchvision

tqdm

torchsummary

## Usage

1. Set up environment:

```bash
conda create -n mnist python=3.8
conda activate mnist
pip install -r requirements.txt
```

2. Train models:

```bash
python train.py
```

3. Monitor training:

- Real-time progress bars
- Epoch-wise metrics
- Model checkpointing

## Model Saving

- Best models are automatically saved as:
  - `best_Model_1.pth`

## Training Features

1. **Progress Tracking**

   - TQDM progress bars
   - Loss and accuracy monitoring
   - Real-time validation metrics
2. **Optimization**

   - Learning rate scheduling
   - Early model checkpointing
   - Device-agnostic training
3. **Evaluation**

   - Training accuracy
   - Validation accuracy
   - Parameter counting
   - Model summary

## Results

- Model_1: Quick baseline (98.63%)

## Future Improvements

1. Add data augmentation
2. Implement cross-validation
3. Add visualization tools
4. Experiment with different optimizers
5. Add model ensemble capabilities

## Acknowledgments

- PyTorch Documentation
- MNIST Dataset
- Deep Learning Community
