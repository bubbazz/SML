import numpy as np
import os

def loadData(path:str):
    X_test = np.loadtxt(os.path.join(path,'mnist_small_test_in.txt'),delimiter=',', dtype =int)
    y_test = np.loadtxt(os.path.join(path,'mnist_small_test_out.txt'),dtype =int)
    X_train = np.loadtxt(os.path.join(path,'mnist_small_train_in.txt'),delimiter=',', dtype =int)
    y_train = np.loadtxt(os.path.join(path,'mnist_small_train_out.txt'),dtype =int)
    return X_train,y_train,X_test,y_test

PATH = os.path.join('.','exc4','dataSets')
X_train,y_train,X_test,y_test = loadData(PATH)
class Network:
    """
    Constructs a neural network according to a list detailing the network structure.

    Args:
        network_structure: List of integers, number of nodes in a layer. The entry network_structure[0]
        equals the number of input features and the last entry is 1 for binary classification.
        network_structure[i] equals the number of hidden units in layer i.
    """

    def __init__(self, network_structure):
        self.num_layers = len(network_structure) - 1
        # state dicts, use integer layer id as keys
        # the range is 0,...,num_layers for x and
        # 1,...,num_layers for all other dicts
        self.w = dict() # weights
        self.b = dict() # biases
        self.z = dict() # outputs of linear layers
        self.x = dict() # outputs of activation layers
        self.dw = dict() # error derivatives w.r.t. w
        self.db = dict() # error derivatives w.r.t. b

        self.init_wb(network_structure)


    def init_wb(self, network_structure):
        for i in range(1, len(network_structure)):
            n_input = network_structure[i - 1]
            n_output = network_structure[i]
            sigma = np.sqrt(1 / n_input)

            self.w[i] = np.random.normal(0, sigma, size=(n_output, n_input))
            self.b[i] = np.zeros(shape=(n_output, 1))


    def sigmoid(self, z):
        """ Sigmoid function.

        Args:
            z: (n_i, b) numpy array, output of the i-th linear layer

        Returns:
            x_out: (n_i, b) numpy array, output of the sigmoid function
        """
        return 1 / (1 + np.exp(-z))


    def sigmoid_backward(self, dx_out, z):
        x_out = self.sigmoid(z)
        dz_in = dx_out * x_out * (1 - x_out)
        return dz_in


    def relu(self, z):
        return np.maximum(0, z)

    def relu_backward(self, dx_out, z):
        dz_in = np.copy(dx_out)
        dz_in[z <= 0] = 0
        return dz_in

    def softmax(self,z):
        return np.exp(z-np.max(z))/sum(np.exp(z))

    def softmax_backward(self,dx_out,z):

        out = np.zeros_like(z)
        sum_classes = np.sum(np.exp(z))
            # z, da shapes - (m, n)
        m, n = z.shape
        p = self.softmax(z)
        tensor1 = np.einsum('ij,ik->ijk', p, p)  # (m, n, n)
        tensor2 = np.einsum('ij,jk->ijk', p, np.eye(n, n))  # (m, n, n)

        dSoftmax = tensor2 - tensor1
        dz = np.einsum('ijk,ik->ij', dSoftmax, dx_out)  # (m, n)
        return dz
        
    def activation_func(self, func, z):
        if func == "sigmoid":
            return self.sigmoid(z)
        elif func == "relu":
            return self.relu(z)
        elif func == "softmax":
            return self.softmax(z)

    def activation_func_backward(self, func, dx_out, z):
        if func == "sigmoid":
            return self.sigmoid_backward(dx_out, z)
        elif func == "relu":
            return self.relu_backward(dx_out, z)
        elif func == "softmax":
            return self.softmax_backward(dx_out, z)


    def layer_forward(self, x_in, func, i):
        self.z[i] = np.dot(self.w[i], x_in) + self.b[i]
        self.x[i] = self.activation_func(func, self.z[i])
        return self.x[i]


    def forward(self, x):
        self.x[0] = x

        x_out = x
        for i in range(1, self.num_layers):
            x_out = self.layer_forward(x_out, 'relu', i)

        predictions = self.layer_forward(x_out, 'softmax', self.num_layers)

        return predictions


    def layer_backward(self, dx_out, func, i):
        b = dx_out.shape[1]

        dz_out = self.activation_func_backward(func, dx_out, self.z[i])
        dx_in = np.dot(self.w[i].T, dz_out)

        self.dw[i] = np.dot(dz_out, self.x[i - 1].T)
        self.db[i] = np.dot(dz_out, np.ones(shape=(1, b)).T)

        return dx_in


    def back_propagation(self, y):

        batch_size = y.shape[1]
        # get predictions from the state dict
        predictions = self.x[self.num_layers]
        # compute the derivative of the mean error regarding the network's output
        d_predictions = - (np.divide(y, predictions) - np.divide(1 - y, 1 - predictions)) / batch_size
        # backward pass through the output layer, updates states of dw and db for the last layer
        dx_in = self.layer_backward(d_predictions, "softmax", self.num_layers)
        # iteratively perform backward propagation through the network layers,
        # update states of dw and db for the i-th layer
        for i in reversed(range(1, self.num_layers)):
            dx_in =  self.layer_backward(dx_in, "relu", i)

        return dx_in


    def update_wb(self, lr):
        """ Update the states of w[i] and b[i] for all layers i based on gradient information
        stored in dw[i] and db[i] and the learning rate.

        Args:
            lr: learning rate
        """
        for i in range(1, self.num_layers + 1):
            self.w[i] -= lr * self.dw[i]
            self.b[i] -= lr * self.db[i]


    def shuffle_data(self, X, Y):
        """ Shuffles the data arrays X and Y randomly. You can use
        np.random.permutation for this method. Make sure that the label
        belonging X_shuffled[:,i] is shuffled to Y_shuffled[:,i].

        Args:
            X: (n_0, B) numpy array, B feature vectors with N_0-dimensional features
            Y: (1, B) numpy array, labels

        Returns:
            X_shuffled: (n_0, B) numpy array, shuffled version of X
            Y_shuffled: (1, B) numpy array, shuffled version of Y
        """
        indices = np.random.permutation(np.arange(X.shape[1]))
        X_shuffled = X[:, indices]
        Y_shuffled = Y[:, indices]

        return X_shuffled, Y_shuffled


    def train(self, X, Y, lr, batch_size, num_epochs):
        """ Trains the neural network with stochastic gradient descent by calling
        shuffle_data once per epoch and forward, back_propagation and update_wb
        per iteration. Start a new epoch if the number of remaining data points
        not yet used in the epoch is smaller than the mini batch size.

        Args: 
            X: (n_0, B) numpy array, B feature vectors with N_0-dimensional features
            Y: (1, B) numpy array, labels
            lr: learning rate
            batch_size: mini batch size for SGD
            num_epochs: number of training epochs
        """
        num_examples = X.shape[1]
        it_per_epoch = num_examples // batch_size
        for i in range(num_epochs):
            X, Y = self.shuffle_data(X, Y)
            print(f"epoch {i}")
            #self.predict(X,Y)
            for i in range(it_per_epoch):
                # extract mini batches
                x = X[:, i * batch_size : (i+1) * batch_size]
                y = Y[:, i * batch_size : (i+1) * batch_size]
                # perform a forward pass, update states of x and z
                _ = self.forward(x)
                # update states of dw and db by performing a backward pass
                _ = self.back_propagation(y)
                # update states of w and b by a SGD step
                self.update_wb(lr)
    def predict(self, X, Y):
        N = len(X)
        pred = np.empty(N)  # Correct predictions

        for i in range(N):
            x = X[i:(i+1), :].T
            out = self.forward(x)
            # print(np.argmax(out), Y[i])
            pred[i] = 1 if np.argmax(out) == Y[i] else 0

        acc = np.sum(pred) / float(N)
        print("ACC = ", acc)
        print("PRED: ", pred.shape)
        print("... ", np.sum((Y == pred)))
        print("--- ", np.sum(pred))
        print("N = ", N)

x = [X_train.shape[0],10,10]
nn = Network(x)
print(y_train.shape)
nn.train(X_train,y_train.reshape(1,-1),0.01,10,10)