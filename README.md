# numpy-neuralnet-exercise

All codes and slides are based on the online book [neuralnetworkanddeeplearning.com](http://neuralnetworksanddeeplearning.com/).

## numpy-only-implementation
example1.py ~ example8.py are implemented via only numpy and use the same architecture of a simple network architecture called multilayer perceptrons (MLP) with one hidden layer.

### example1.py : Quadratic Loss, SGD, Sigmoid (BASE LINE)
### example2.py : BASE LINE + Cross Entropy Loss
### example3.py : BASE LINE + Cross Entropy Loss + L2 regularization
### example4.py : BASE LINE + Cross Entropy Loss + L1 regularization
### example5.py : BASE LINE + Cross Entropy Loss + Droput
### example6.py : BASE LINE + Cross Entropy Loss + Xavier Initializer
### example7.py : BASE LINE + Cross Entropy Loss + Momentum based SGD
### example8.py : BASE LINE + Cross Entropy Loss + Xavier Initializer + ReLU

## lauchers for other resources of numpy-only-implementation  
There are also good resources for numpy-only-implementation and laucher for each recourse is provided.

*Resource* | *Launcher*
:---: | :---: |
[neuralnetworkanddeeplearning.com](https://github.com/mnielsen/neural-networks-and-deep-learning) | [launcher_package1.py](https://github.com/hwalsuklee/numpy-neuralnet-exercise/blob/master/launcher_package1.py) |   
[Stanford CS231 lectures](https://github.com/cthorey/CS231/tree/master/assignment2) | [launcher_package2.py](https://github.com/hwalsuklee/numpy-neuralnet-exercise/blob/master/launcher_package2.py) |  

## simple tensoflow code for CNN
Code in tf_code_mnist folder is for CNN implmentation.  
[ch6_summary.pdf](https://github.com/hwalsuklee/numpy-neuralnet-exercise/blob/master/slides/ch6_summary.pdf) is related slide.

### train --model v0 : BASE LINE + Softmax Layer + Cross Entropy Loss
### train --model v1 : model v0 + 1 Convolutional/Pooling Layers
### train --model v2 : model v1 + 1 Convolutional/Pooling Layers
### train --model v3 : model v2 + ReLU
### train --model v4 : model v3 + Data Augmentation
### train --model v5 : model v4 + 1 Fully-Connected Layer
### train --model v6 : model v5 + Dropout
