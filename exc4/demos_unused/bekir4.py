import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:

    def __init__(self, network_structure):
        # Key: layer id [1..]
        self.weights = dict()
        self.biases = dict()
        self.out_activation = dict()            # 0: input pixels, [1..]: f(Wa + b)
        self.out_layer = dict()                 # Wa + b
        self.weights_derivative_err = dict()
        self.biases_derivative_err = dict()

        self.num_layers = len(network_structure) - 1
        for i in range(1, self.num_layers + 1):
            self.weights[i] = np.random.randn(network_structure[i], network_structure[i - 1]) / np.sqrt(network_structure[i - 1])
            self.biases[i] = np.zeros((network_structure[i], 1))

    def relu(self, out_layer):
        # return np.array([np.max(0, out_layer[i]) for i in range(len(out_layer))])
        return np.maximum(np.zeros_like(out_layer), out_layer)

    def relu_backward(self, out_layer, dx_out):
        output = np.array(dx_out)
        output[out_layer <= 0] = 0
        return output

    def softmax(self, out_layer):
        # print(out_layer)
        out = np.empty_like(out_layer)
        sum_classes = np.sum(np.exp(out_layer))     # TODO: can cause overflow -> results in nan values (likely because weight updates are wrong)
        for i in range(len(out)):
            out[i] = np.exp(out_layer[i]) / sum_classes
        return out

    # https://www.mldawn.com/the-derivative-of-softmaxz-function-w-r-t-z/
    def softmax_backward(self, out_layer, dx_out):
        # print("SSS: ", dx_out)
        out = np.zeros_like(out_layer)
        sum_classes = np.sum(np.exp(out_layer))
        # This loop is nonsense
        for i in range(len(out_layer)):
            for j in range(len(out_layer)):
                if i == j:
                    out[i] = (np.exp(out_layer[i]) / sum_classes) * (1 - (np.exp(out_layer[i]) / sum_classes))
                    # out[i] = out_layer[i] * (1 - out_layer[i])
                else:
                    out[i] = -(np.exp(out_layer[i]) / sum_classes) * (np.exp(out_layer[j]) / sum_classes)
                    # out[i] = -out_layer[i] * out_layer[j]
        #print( out)
        return dx_out * out  # ~= 0

    def activation(self, out_layer, strat):
        if strat == "relu":
            return self.relu(out_layer)
        else:
            return self.softmax(out_layer)

    def activation_backward(self, out_layer, dx_out, strat):
        if strat == "relu":
            return self.relu_backward(out_layer, dx_out)
        else:
            return self.softmax_backward(out_layer, dx_out)

    def layer_forward(self, x, layer_id, strat):
        w = self.weights[layer_id]
        b = self.biases[layer_id]
        # print("W: ", w.shape, " -- b: ", b.shape)
        self.out_layer[layer_id] = (w @ x) + b
        self.out_activation[layer_id] = self.activation(self.out_layer[layer_id], strat)
        return self.out_activation[layer_id]

    def forward(self, inp):
        # Output of input layer (id 0): x_i
        self.out_activation[0] = inp

        # Use ReLU in-between
        for i in range(1, self.num_layers):
            inp = self.layer_forward(inp, i, "relu")

        # Use softmax for output
        return self.layer_forward(inp, self.num_layers, "softmax")

    def layer_backward(self, dx_out, layer_id, strat):
        w = self.weights[layer_id]
        res = self.activation_backward(self.out_layer[layer_id], dx_out, strat)  # Eg. dx_out == dpred first time

        self.weights_derivative_err[layer_id] = np.dot(res, self.out_activation[layer_id - 1].T)
        self.biases_derivative_err[layer_id] = np.sum(res, axis=1, keepdims=True)

        return np.dot(w.T, res)

    def backward(self, y):
        # Perhaps this is better:
        pred = self.out_activation[self.num_layers]  # (10, 1) ... % for 0-9
        targ = np.zeros_like(pred)
        targ[int(y[0][0])] = 1
        dpred = targ - pred

        # The output of the forward pass (TODO: can be nan... @see softmax)
        # Predict the entry with the highest probability
        pred = np.argmax(self.out_activation[self.num_layers])

        # print("P: ", pred.shape, ", Y: ", y.shape, " ... ", (pred - y).shape)
        # print("P: ", pred, ", Y: ", y, " ... ", (pred - y))

        # Use cross-entropy loss (https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html)
        # Derivative is simply (prediction - target) ?
        # dpred = (pred - y)[0][0]    # For some reason this has the form [[x]]

        dx_in = self.layer_backward(dpred, self.num_layers, "softmax")

        for i in reversed(range(1, self.num_layers)):
            dx_in = self.layer_backward(dx_in, i, "relu")

        return dx_in

    def update_wb(self, learning_rate):
        for i in range(1, self.num_layers + 1):
            self.weights[i] -= learning_rate * self.weights_derivative_err[i]
            self.biases[i] -= learning_rate * self.biases_derivative_err[i]

    def shuffle_data(self, X, Y):
        indices = np.random.permutation(np.arange(X.shape[0]))
        X = X[indices, :]
        Y = Y[indices, :]
        return X, Y

    def train(self, X, Y, learning_rate):
        num_examples = X.shape[0]

        X, Y = self.shuffle_data(X, Y)
        for epoch in range(50):
            print("Epoch: ", epoch)
            if epoch % 5 == 0:
                self.pred(X,Y)
            for i in range(num_examples):
                # extract mini batches
                x = X[i:(i + 1), :]
                y = Y[i].squeeze().reshape(-1, 1)
                # print(x.shape, y.shape)
                # perform a forward pass, update states of x and z
                _ = self.forward(x.T)
                # TODO: cost/loss function?
                # update states of dw and db by performing a backward pass
                _ = self.backward(y)
                # update states of w and b by a SGD step
                self.update_wb(learning_rate)

    def pred(self, X, Y):
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


def main():
    np.random.seed(38)

    # 28x28 == 784;
    small_train_in = np.loadtxt("exc4/dataSets/mnist_small_train_in.txt", delimiter=",")     # (6006, 784)
    small_train_out = np.loadtxt("exc4/dataSets/mnist_small_train_out.txt")                  # (6006, )
    small_test_in = np.loadtxt("exc4/dataSets/mnist_small_test_in.txt", delimiter=",")       # (1004, 784)
    small_test_out = np.loadtxt("exc4/dataSets/mnist_small_test_out.txt")                    # (1004, )

    small_train_out = small_train_out.reshape((small_train_out.shape[0], 1))
    small_test_out = small_test_out.reshape((small_test_out.shape[0], 1))

    print(small_train_in.shape, small_train_out.shape)
    print(small_test_in.shape, small_test_out.shape)

    # TODO implement neural network using backpropagation
    # Choose loss and activation functions
    # Choose hyperparameters (number of layers, neurons, learning rate, ...)
    # Optional: gradient descent optimizer
    # -> Goal: misclassification error < 8% on testing set
    net_structure = [784, 16, 10]
    lr = 0.01
    nn = NeuralNetwork(net_structure)
    nn.train(small_train_in, small_train_out, lr)

    # Now 'nn' has learned parameters w and b
    nn.pred(small_train_in, small_train_out)
    # nn.pred(small_test_in, small_test_out)

    print("END")


if __name__ == "__main__":
    main()

