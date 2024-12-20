# MNIST Classification with PyTorch

A lightweight CNN architecture designed to achieve >99.4% accuracy on MNIST with less than 8,000 parameters.

## Model Architecture

Model_4 Architecture:

---

Layer (type) Output Shape Param #

================================================================

Conv2d-1 [-1, 6, 28, 28] 54

BatchNorm2d-2 [-1, 6, 28, 28] 12

ReLU-3 [-1, 6, 28, 28] 0

Dropout-4 [-1, 6, 28, 28] 0

Conv2d-5 [-1, 12, 28, 28] 648

BatchNorm2d-6 [-1, 12, 28, 28] 24

ReLU-7 [-1, 12, 28, 28] 0

Dropout-8 [-1, 12, 28, 28] 0

MaxPool2d-9 [-1, 12, 14, 14] 0

Conv2d-10 [-1, 24, 14, 14] 2,592

BatchNorm2d-11 [-1, 24, 14, 14] 48

ReLU-12 [-1, 24, 14, 14] 0

Dropout-13 [-1, 24, 14, 14] 0

Conv2d-14 [-1, 12, 14, 14] 2,592

BatchNorm2d-15 [-1, 12, 14, 14] 24

ReLU-16 [-1, 12, 14, 14] 0

Dropout-17 [-1, 12, 14, 14] 0

MaxPool2d-18 [-1, 12, 7, 7] 0

Conv2d-19 [-1, 10, 7, 7] 1,080

BatchNorm2d-20 [-1, 10, 7, 7] 20

ReLU-21 [-1, 10, 7, 7] 0

Dropout-22 [-1, 10, 7, 7] 0

AdaptiveAvgPool2d-23 [-1, 10, 1, 1] 0

Conv2d-24 [-1, 10, 1, 1] 100

================================================================

Total params: 7,204

Trainable params: 7,204

Non-trainable params: 0
