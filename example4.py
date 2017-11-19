#############################################################
###   BASE LINE + Cross Entropy Loss  + L1 Regularization ###
#############################################################

import numpy as np
import mnist_loader

### Data Loading

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

### Parameters

n_epoch = 30
learning_rate = 0.5
batch_size = 10
lamda = 5

### Network Architecture

n_node_input = 784
n_node_hidden = 30
n_node_output = 10

### Weight & Bias

W2=np.random.randn(n_node_hidden, n_node_input)
b2=np.random.randn(n_node_hidden, 1)

W3=np.random.randn(n_node_output, n_node_hidden)
b3=np.random.randn(n_node_output, 1)

### Activation Functions

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

### Training
test_errors = []
training_errors = []
n = len(training_data)

file_name_common = 'l1reg_ce'+'_nHidden'+str(n_node_hidden)+'.txt'

try:
    training_errors = np.loadtxt(fname='tr_'+file_name_common)
    test_errors = np.loadtxt(fname='test_'+file_name_common)
except:
    for j in range(n_epoch):

        ## Stochastic Gradient Descent
        np.random.shuffle(training_data)

        # for each batch
        sum_of_training_error = 0
        for k in range(0, n, batch_size):
            batch = training_data[k:k+batch_size]

            # average gradient for samples in a batch
            sum_gradient_b3 = 0
            sum_gradient_b2 = 0
            sum_gradient_W3 = 0
            sum_gradient_W2 = 0

            # for each sample
            for x, y in batch:
                ## Feed forward

                a1 = x
                z2 = np.dot(W2, a1) + b2
                a2 = sigmoid(z2)
                z3 = np.dot(W3, a2) + b3
                a3 = sigmoid(z3)

                ## Backpropagation

                # Step 1: Error at the output layer [Cross-Entropy Cost]
                delta_3 = (a3-y)
                # Step 2: Error relationship between two adjacent layers
                delta_2 =  sigmoid_prime(z2)*np.dot(W3.transpose(), delta_3)
                # Step 3: Gradient of C in terms of bias
                gradient_b3 = delta_3
                gradient_b2 = delta_2
                # Step 4: Gradient of C in terms of weight
                gradient_W3 = np.dot(delta_3, a2.transpose())
                gradient_W2 = np.dot(delta_2, a1.transpose())

                # update gradients
                sum_gradient_b3 += gradient_b3
                sum_gradient_b2 += gradient_b2

                sum_gradient_W3 += gradient_W3
                sum_gradient_W2 += gradient_W2

                ## Training Error
                sum_of_training_error += int(np.argmax(a3) != np.argmax(y))

            # Update Biases
            b3 -= learning_rate * sum_gradient_b3 / batch_size
            b2 -= learning_rate * sum_gradient_b2 / batch_size

            # Update Weights
            # L1 regularization
            W3 -= (learning_rate * lamda / n)*np.sign(W3)
            W2 -= (learning_rate * lamda / n)*np.sign(W2)
            # update
            W3 -= learning_rate * sum_gradient_W3 / batch_size
            W2 -= learning_rate * sum_gradient_W2 / batch_size

        # Report Training Error
        print("[TRAIN_ERROR] Epoch %02d: %5d / %05d" % (j, sum_of_training_error, n))
        training_errors.append(np.float(sum_of_training_error) / n)

        ### Test
        n_test = len(test_data)
        sum_of_test_error = 0
        for x, y in test_data:
            ## Feed forward

            a1 = x
            z2 = np.dot(W2, a1) + b2
            a2 = sigmoid(z2)
            z3 = np.dot(W3, a2) + b3
            a3 = sigmoid(z3)

            ## Test Error
            # in test data, label info is a number not one-hot vector as in training data
            sum_of_test_error += int(np.argmax(a3) != y)

        # Report Test Error
        print("[ TEST_ERROR] Epoch %02d: %5d / %05d" % (j, sum_of_test_error, n_test))

        test_errors.append(np.float(sum_of_test_error)/n_test)

    ## Save Results
    np.savetxt('tr_'+file_name_common, np.array(training_errors), fmt='%.5f')
    np.savetxt('test_'+file_name_common, np.array(test_errors), fmt='%.5f')

### Plot results
import matplotlib.pyplot as plt
idx = np.arange(1,n_epoch+1)

plt.plot(idx, np.array(test_errors)*100,'ro-', label='with L1 regularization')
try:
    # Load baseline
    file_name_common = 'ce'+'_nHidden'+str(n_node_hidden)+'.txt'
    mse = np.loadtxt(fname='test_'+file_name_common)
    plt.plot(idx,np.array(mse)*100,'bo-', label='without L1 regularization')
except:
    print ('There is no result of baseline')

legend = plt.legend(loc='upper center', shadow=True)
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}
plt.rc('font', **font)
plt.xlabel('Epoch', fontsize=22)
plt.ylabel('Test error rate [%]', fontsize=22)
plt.grid(True)
plt.show()