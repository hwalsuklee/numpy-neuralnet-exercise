# numpy-neuralnet-exercise
numpy와 mnist만을 가지고 뉴럴넷에서 사용된 기본 개념들을 구현해봅니다.

## 목차
모든 네트워크 구조는 fully-connect layer 기반에 hidden-layer가 하나만 있는 구조입니다.

### example1.py : Quadratic Loss, SGD, Sigmoid (BASE LINE)
### example2.py : BASE LINE + Cross Entropy Loss
### example3.py : BASE LINE + Cross Entropy Loss + L2 regularization
### example4.py : BASE LINE + Cross Entropy Loss + L1 regularization
### example5.py : BASE LINE + Cross Entropy Loss + Droput
### example6.py : BASE LINE + Cross Entropy Loss + Xavier Initializer
### example7.py : BASE LINE + Cross Entropy Loss + Momentum based SGD
### example8.py : BASE LINE + Cross Entropy Loss + Xavier Initializer + ReLU

## Open Package Launcher
### Pakcage 1 (https://github.com/mnielsen/neural-networks-and-deep-learning)
### Pakcage 2 (https://github.com/cthorey/CS231/tree/master/assignment2)

## Tensorflow version
tf_code_mnist 폴더 내의 소스 코드 사용.

### train --model v0 : BASE LINE + Softmax Layer + Cross Entropy Loss
### train --model v1 : model v0 + 1 Convolutional/Pooling Layers
### train --model v2 : model v1 + 1 Convolutional/Pooling Layers
### train --model v3 : model v2 + ReLU
### train --model v4 : model v3 + Data Augmentation
### train --model v5 : model v4 + 1 Fully-Connected Layer
### train --model v6 : model v5 + Dropout
