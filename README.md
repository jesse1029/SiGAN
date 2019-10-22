# SiGAN

The implementation of paper Chih-Chung Hsu, Chia-Wen Lin, Weng-Tai Su, Gene Cheung, SiGAN: Siamese Generative Adversarial Network for Identity-Preserving Face Hallucination, published in IEEE Transactions on Image Processing (TIP) 2019.
Please cite if you adopt our code on your research.

We modify the code forked from https://github.com/david-gpu/srez to implement pairwise learning architecture for face hallucination.

# Reqirements
Tensorflow 1.13~ 1.08. Not support tensorflow 2.0 yet.

# Pretrained Model
The trained model for super-resolve 32x32 to 128x128 image can be downloaded from
https://drive.google.com/file/d/1qvWqsRfP2hZrZzXOG4NmZRuxb7fHkFAe/view?usp=sharing

# How to use
## Recommend: Use Jupyter Notebook env.
## Training with your own dataset:

srez_train_sia.ipynb

## Testing your own image with trained model

test_sia.ipynb

# Dataset
We train the model on CASIA-WebFace dataset.
