# Package 2
# from https://github.com/cthorey/CS231/tree/master/assignment2

import numpy as np
import mnist_loader

import matplotlib.pyplot as plt
from cs231n.classifiers.fc_net import *
from cs231n.solver import Solver


### Data Loading

training_data, validation_data, test_data = mnist_loader.load_data_wrapper(augmentation=False)


X_train, y_train = zip(*training_data)
X_val, y_val = zip(*test_data)

X_train = np.array(X_train)
y_train = np.array(np.squeeze(np.argmax(y_train, axis=1)))
X_val = np.array(X_val)
y_val = np.array(y_val)

### Parameters

n_epoch = 30
learning_rate = 0.001
batch_size = 10

### Network Architecture

n_node_input = 784
hidden_dims = [100]
n_node_output = 10

""""""""""""""""""""""""""""""""""""
""" Batchnorm for deep networks  """
""""""""""""""""""""""""""""""""""""
# Try training a very deep net with batchnorm
hidden_dims = [100]

full_data = {
  'X_train': X_train,
  'y_train': y_train,
  'X_val': X_val,
  'y_val': y_val,
}

weight_scale = 2e-2 # this is for weight-initializer
bn_model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=True, input_dim=n_node_input)
model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=False, input_dim=n_node_input)

bn_solver = Solver(bn_model, full_data,
                num_epochs=n_epoch, batch_size=batch_size,
                update_rule='adam',
                optim_config={
                  'learning_rate': learning_rate,
                },
                verbose=True, print_every=200)
bn_solver.train()

solver = Solver(model, full_data,
                num_epochs=n_epoch, batch_size=batch_size,
                update_rule='adam',
                optim_config={
                  'learning_rate': learning_rate,
                },
                verbose=True, print_every=200)
solver.train()

### Plot Results
#matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

plt.subplot(2, 1, 1)
plt.title('Training accuracy')
plt.xlabel('Iteration')

plt.subplot(2, 1, 2)
plt.title('Validation accuracy')
plt.xlabel('Epoch')

plt.subplot(2, 1, 1)
plt.plot(solver.train_acc_history, '-o', label='baseline')
plt.plot(bn_solver.train_acc_history, '-o', label='batchnorm')

plt.subplot(2, 1, 2)
plt.plot(solver.val_acc_history, '-o', label='baseline')
plt.plot(bn_solver.val_acc_history, '-o', label='batchnorm')

for i in [1, 2]:
    plt.subplot(2, 1, i)
    plt.legend(loc='upper center', ncol=4)
plt.gcf().set_size_inches(15, 15)
plt.show()