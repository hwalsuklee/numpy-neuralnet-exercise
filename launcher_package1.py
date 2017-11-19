# Package 1
# from https://github.com/mnielsen/neural-networks-and-deep-learning

import layer
import mnist_loader

### Data Loading

training_data, validation_data, test_data = mnist_loader.load_data_wrapper(augmentation=False)

### Parameters

n_epoch = 3
learning_rate = 0.5
batch_size = 50

### Network Architecture

n_node_input = 784
n_node_hidden = 30
n_node_output = 10

net = layer.Network([n_node_input, n_node_hidden, n_node_output],
                       W_init='xavier',             # or normal
                       b_init='zero',               # or normal
                       cost=layer.CrossEntropyCost, # or QuadraticCost
                       act_fn=layer.Sigmoid         # or Relu
                       )

### Training

# SGD
evaluation_cost, evaluation_accuracy, \
training_cost, training_accuracy = \
net.SGD(training_data, n_epoch, batch_size, learning_rate,
        lmbda=0.0,                          # L2 regularization
        evaluation_data=test_data,
        monitor_evaluation_cost=False,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=False,
        monitor_training_accuracy=True)

### Plot results
import matplotlib.pyplot as plt
import numpy as np

idx = np.arange(1,n_epoch+1)

plt.plot(idx, evaluation_accuracy,'ro-', label='test acc.')
plt.plot(idx, training_accuracy,'bo-', label='training acc.')

legend = plt.legend(loc='upper center', shadow=True)
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}
plt.rc('font', **font)
plt.xlabel('Epoch', fontsize=22)
plt.ylabel('Accuracy [%]', fontsize=22)
plt.grid(True)
plt.show()