---
name: Model proposal
about: Suggest a deep learning model to test
title: ''
labels: ''
assignees: ''

---

**Model architecture.**
Ex.

Layer | Type | Size | Features |  Kernel | Stride | Activation | Notes
--------|--------|-------|-------------|-----------|---------|-------------|---------
In | Input | 69x69 | 3 | | | | Input Image
1 | Conv2D | 64x64 | 32 | 6x6 | 1 | ReLU |
2 | MaxPool2D | 32x32 | 32 | 2x2 | 2 | |
3 | Dense | 32,768 | | | | | Flattened layer
4 | Dense | 64 | | | | ReLU |
Out | Dense | 3 | | | | Softmax | Output prediction

**Hyperparameters**
Optimizer: e.g. Adam, AdaGrad
Loss: e.g. categorical_crossentropy, binary_crossentropy, categorical_hinge
Metrics: e.g. accuracy, loss

**Describe alternative hyperparameters**
Any other hyperparameter settings you have considered

**Additional context**
Add any other context or information about the proposed model here.

**References**
List any relevant references.
