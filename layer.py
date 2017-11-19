# From https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network2.py

# Third-party libraries
import numpy as np

#### Define non-linear activation functions
class Sigmoid(object):
    @staticmethod
    def activation(z):
        return 1.0/(1.0+np.exp(-z))

    @staticmethod
    def prime(z):
        z = 1.0/(1.0+np.exp(-z))
        return z*(1-z)

class Relu(object):
    @staticmethod
    def activation(z):
        return np.maximum(z, 0)

    @staticmethod
    def prime(z):
        return np.where(z > 0, 1.0, 0.0)

#### Define the quadratic and cross-entropy cost functions
class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(act_fn_prime_z, a, y):
        return (a-y) * act_fn_prime_z


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(act_fn_prime_z, a, y):
        return (a-y)


#### Main Network class
class Network(object):

    def __init__(self, sizes,
                 W_init='xavier',
                 b_init='zero',
                 cost=CrossEntropyCost,
                 act_fn=Relu):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weight_initializer(W_init)
        self.bias_initializer(b_init)
        self.cost=cost
        self.act_fn=act_fn

    def weight_initializer(self, W_init):
        if W_init == 'xavier':
            self.weights = [np.random.randn(y, x) / np.sqrt(x)
                            for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        elif W_init == 'he':
            self.weights = [np.random.randn(y, x) / np.sqrt(x/2)
                            for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        elif W_init == 'normal':
            self.weights = [np.random.randn(y, x)
                            for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        else:
            NotImplementedError()

    def bias_initializer(self, b_init):
        if b_init == 'normal':
            self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        elif b_init == 'zero':
            self.biases = [np.zeros([y, 1]) for y in self.sizes[1:]]
        else:
            NotImplementedError()


    def feedforward(self, a):
        for b, W in zip(self.biases, self.weights):
            a = self.act_fn.activation(np.dot(W, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):

        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
            print("Epoch %s training complete" % j)
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy/np.float(n))
                print("Accuracy on training data: {} / {}".format(
                    accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy/np.float(n_data))
                print("Accuracy on evaluation data: {} / {}".format(
                    self.accuracy(evaluation_data), n_data))
            print
        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        x, y = zip(*mini_batch)
        x = np.squeeze(x)
        y = np.squeeze(y)

        nabla_b, nabla_w = self.backprop(x, y)

        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, W in zip(self.biases, self.weights):
            z = np.dot(activation, W.transpose()) + b.transpose()
            zs.append(z)
            activation = self.act_fn.activation(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(self.act_fn.prime(zs[-1]), activations[-1], y)
        nabla_b[-1] = np.expand_dims(np.sum(delta,axis=0),axis=1)
        nabla_w[-1] = np.dot(delta.transpose(),activations[-2])
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.act_fn.prime(z)
            delta = sp * np.dot(delta, self.weights[-l+1])
            nabla_b[-l] = np.expand_dims(np.sum(delta,axis=0),axis=1)
            nabla_w[-l] = np.dot(delta.transpose(), activations[-l-1])
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

#### Miscellaneous functions
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e