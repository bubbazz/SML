# SML Exercise 4 (Group 38: Javor and Bekir)

## Task 1: Neural Networks

**Important**: there was a NN task in the last semester of Computer Vision 1. We have taken that as a basis.

Our network has one input layer, one hidden layer, and one output layer. 
For the input layer there is one neuron per pixel (784 in total). The output layer has one neuron per class (10 for the numbers 0, 1, 2, ..., 8, 9). The number of layers for the hidden layer was arbitrarily chosen to be 16.
```
net_structure = [784, 16, 10]
nn = NeuralNetwork(net_structure)
```

We train the network using individual examples (ie. no batches). We set the learning rate to `1e-5`. We had to set it reasonably small because faster learning rates led to stability problems (eg. overflows during the computation).

```
for epoch in range(50):
    print("Epoch: ", epoch)
    acc.append(self.pred(X, Y))
    for i in range(num_examples):
        x = X[i:(i + 1), :]
        y = Y[i].squeeze().reshape(-1, 1)
        _ = self.forward(x.T)
        _ = self.backward(y)
        self.update_wb(learning_rate)
```

For the forward step we use ReLU activations for the in-between layers and `softmax` for the output layer.
```
def forward(self, inp):
    # Output of input layer (id 0): x_i
    self.out_activation[0] = inp

    # Use ReLU in-between
    for i in range(1, self.num_layers):
        inp = self.layer_forward(inp, i, "relu")

    # Use softmax for output
    return self.layer_forward(inp, self.num_layers, "softmax")
```
The reason for using ReLU is simply because many sources (including the slides) state that is a good choice in general. The choice of `softmax` is not quite as arbitrary. What we want as output is some sort of probabilities as to what the network thinks the current output should be. Using `sigmoid` we could obtain values in [0,1] for each of the neurons, but we also have to ensure that all probabilities sum up to one. That's what `softmax` does, so it's a suitable choice.

The forward step:
```
def layer_forward(self, x, layer_id, strat):
    w = self.weights[layer_id]
    b = self.biases[layer_id]
    self.out_layer[layer_id] = (w @ x) + b
    self.out_activation[layer_id] = self.activation(self.out_layer[layer_id], strat)
    return self.out_activation[layer_id]
```
We store both the linear output $\sum w_ix_i + b$ as well as the activation $a(\sum w_ix_i + b)$ thereof, where $a()$ is either ReLU or softmax (determined by the input argument `strat`).

The implementation of ReLU and softmax are straightforward:
```
def relu(self, out_layer):
    # out_layer is the linear output 
    return np.maximum(np.zeros_like(out_layer), out_layer)


# http://www.adeveloperdiary.com/data-science/deep-learning/neural-network-with-softmax-in-python/
def softmax(self, out_layer):
    sum_classes = np.sum(np.exp(out_layer) - np.max(out_layer))
    return np.exp(out_layer - np.max(out_layer)) / sum_classes
```
The softmax function is modified slightly to be more stable by multiplying the numerator and denumerator with `c = -max(z)` (see http://www.adeveloperdiary.com/data-science/deep-learning/neural-network-with-softmax-in-python/).


The backward step:
```
def backward(self, y):
    pred = self.out_activation[self.num_layers]
    targ = np.zeros_like(pred)
    targ[int(y[0][0])] = 1
    dpred = targ - pred

    dx_in = self.layer_backward(dpred, self.num_layers, "softmax")

    for i in reversed(range(1, self.num_layers)):
        dx_in = self.layer_backward(dx_in, i, "relu")

    return dx_in
```
We use cross-entropy (multiclass) as our loss function, whose derivative is simply $p - t$, where $t$ is one-hot encoded [see https://peterroelants.github.io/posts/cross-entropy-softmax/].

The activations for the backward step:
```
def relu_backward(self, out_layer, dx_out):
    output = np.array(dx_out)
    output[out_layer <= 0] = 0
    return output

# https://www.mldawn.com/the-derivative-of-softmaxz-function-w-r-t-z/
# https://themaverickmeerkat.com/2019-10-23-Softmax/
def softmax_backward(self, out_layer, dx_out):
    z = out_layer
    out = np.zeros_like(z)
    sum_classes = np.sum(np.exp(z))
    m, n = z.shape
    p = self.softmax(z)
    tensor1 = np.einsum('ij,ik->ijk', p, p)  # (m, n, n)
    tensor2 = np.einsum('ij,jk->ijk', p, np.eye(n, n))  # (m, n, n)

    dSoftmax = tensor2 - tensor1
    out = np.einsum('ijk,ik->ij', dSoftmax, dx_out)  # (m, n)
    return dx_out * out  
```
These are the derivatives of the ReLU and softmax functions, respectively. ReLU is again staightforward ($1$ if $x > 0$, else $0$). 
Softmax is a bit more involved. We've tried our own versions, but in the end decided to go with (https://themaverickmeerkat.com/2019-10-23-Softmax/). It didn't help either way.

Note that for softmax we multiply the derivative by our output error (which is passed in as the `dx_out` argument).

Alltogether, these functions are used to propagate the error back along the network.
```
def layer_backward(self, dx_out, layer_id, strat):
    w = self.weights[layer_id]
    res = self.activation_backward(self.out_layer[layer_id], dx_out, strat) 

    self.weights_derivative_err[layer_id] = np.dot(res, self.out_activation[layer_id - 1].T)
    self.biases_derivative_err[layer_id] = np.sum(res, axis=1, keepdims=True)

    return np.dot(w.T, res)
```
For each step (layer) we store the errors for the weights and biases. These are then used to update the weights and biases accordingly:
```
def update_wb(self, learning_rate):
    for i in range(1, self.num_layers + 1):
        self.weights[i] -= learning_rate * self.weights_derivative_err[i]
        self.biases[i] -= learning_rate * self.biases_derivative_err[i]
```

To actually do predictions, all we have to do is a forward pass and take the neuron of the output layer with the largest probability:
```
def pred(self, X, Y):
    N = len(X)
    pred = np.empty(N)  # Correct predictions

    for i in range(N):
        x = X[i:(i+1), :].T
        out = self.forward(x)
        pred[i] = 1 if np.argmax(out) == Y[i] else 0

    acc = np.sum(pred) / float(N)
```

#### Discussion

This is the accuracy of our network after 50 iterations.

Train set:
![](https://i.imgur.com/OZrWNU8.png)

Test set:
![](https://i.imgur.com/lUh1cWt.png)


Unfortunately our network is basically just guessing. To make matters worse, its performance gradually worsens, and for the test set it actually stagnates after a certain point, which we cannot explain. 
The culprit is likely the backward step, probably a wrong update of the weight/bias derivatives. We weren't able to fix this problem, however. 

Another thing that came to mind (since the curve is going down) was to change the weight/bias update to use a `+=` instead of a `-=` (although in our opinion this should not be, unless there's a sign error somewhere else in the code). Even so, the network started degrading after an initial improvement.
![](https://i.imgur.com/NicPcJL.png)

Unfortunately we were not able to test this out on more (as in thousands) iterations because our implementation is slow. It takes about 10 minutes for 50 iterations.

Even so, 50 iterations should still be enough to see some significant change instead of roaming around the 10% mark, which is essentially just guessing.

## Task 2: Support Vector Machines

Source: MML-Book (Chapter 12)

#### a

SVMs solve the binary classification task, ie. given a dataset with two classes, it finds parameters that allow us to predict classes on before unseen values.
They do so by finding a hyperplane that separates the two classes along with a margin on both sides.

The primary advantage of this approach is that the margin allows it to better deal with examples that are close to the "intersection" of both classes.

#### b

The constrained optimization problem can be formulated as follows: 

$min \tfrac{1}{2} ||w||^2$ subject to $y_n(\vec{w}_n \cdot \vec{x}_n + b) \ge 1$

#### c

Slack variables allow us to deal with cases that are not (perfectly) linearly separable because some points lie within the margin or are on the wrong side of the hyperplane.

Below, $C > 0$ is a constant and $\xi_n \ge 0$ are the slack variabels. This is called the soft margin:

$min \tfrac{1}{2} ||w||^2 + C \cdot \Sigma_i^N \xi_n$ subject to $y_n(\vec{w}_n \cdot \vec{x}_n + b) \ge 1 - \xi_n$

#### d
so get the maximum we need to derivate the Lagrancian and set it to zero:
\begin{align}
\frac{\partial L}{\partial w} &= \partial_w( |w|^2 + C\sum^N_{n=1} \xi_n -\sum^N_{n=1} a_n(t_n ( w \phi(x_n)+b)-1+\xi_n)-\sum^N_{n=1} \mu_n\xi_n)\\
&= w -\sum^N_{n=1} a_n(t_n\phi(x_n)) = 0
\end{align}
\begin{align}
\frac{\partial L}{\partial b} &= \partial_b( |w|^2 + C\sum^N_{n=1} \xi_n -\sum^N_{n=1} a_n(t_n (w \phi(x_n)+b)-1+\xi_n)-\sum^N_{n=1} \mu_n\xi_n)\\
&= -\sum^N_{n=1} a_nt_n = 0
\end{align}
\begin{align}
\frac{\partial L}{\partial \xi_n} &= \partial_{\xi_n}( |w|^2 + C\sum^N_{n=1} \xi_n -\sum^N_{n=1} a_n(t_n (w \phi(x_n)+b)-1+\xi_n)-\sum^N_{n=1} \mu_n\xi_n)\\
&=C -a_n - \mu_n = 0
\end{align}
\begin{align}
a \ge 0\\
\mu_n\ge 0\\
\xi_n \ge 0 \\
\end{align}
as mentioned if $a_n = 0$ its not a support vector but there is more constrains to idenitfy
if $C = a_n$ the other lagranian need to be zero $\mu_n = 0$ therforce the slackvariable could be greater then zero and the point could wrong classified 
is $C > a_n $ so $\mu_n$ is greater then zero so the slackvariable need to be zero to minimize the equation. analog to the normal SVM Problem this point will be a suport vector.

b need to be calculate from condtional equation $t_ny(x_n)= 1 -\xi_n$ where mention before if $ C > a_n > 0$ $\xi_n$ need to be zero so we simplify the equation 
\begin{align}
t_n(\sum_{m \in SV(X)} a_mt_m\phi(x_n)\phi(x_m) +b )= 1 \\
b = t_n - \sum_{m \in SV(X)}a_mt_m\phi(x_n)\phi(x_m) 
\end{align}
we can also average b over all SV
#### e

For dual SVMs, the number of parameters increase with the number of examples $N$ instead of the number of features $D$. So, in terms of complexity they are more useful than primal SVMs if $D > N$, ie. if we have more features than examples.

Kernels can easily be applied, which implies further advantages, eg. the ability to work with high or infinite dimensional feature spaces (Bishop, Ch. 6.1).

We obtain $\alpha_i$ which are non-zero only for the support vectors. This implies computational efficiency as opposed to calculating $\vec{w} \cdot \vec{x}$ if there are only a few support vectors (https://peterroelants.github.io/posts/cross-entropy-softmax/).

#### f

The kernel trick is a generalization from the inner product ($\vec{x_i} \cdot \vec{x_j}$) to a kernel function ($\kappa(\vec{x_i}, \vec{x_j}$)). It hides away the explicit non-linear feature map. The kernel is often written so that it is computationally more efficient than the calculation of the inner product (especially when considering high dimensions). With kernels we can easily apply our SVM to different kinds of objects, eg. strings and graphs.

#### g

The SVM has a linear Kernel with 

```python=
import os
import cvxopt
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix as cvxopt_matrix
##
from sklearn.svm import SVC

PATH = os.path.join('.','exc4','dataSets',"iris-pca.txt")
data = np.loadtxt(PATH)

X,y = data[:,0:2],data[:,2]
def y_classifier(y):
    return np.array([1 if yi == 2 else -1 for yi in y ])

def SVM_normal(X:np.array,y:np.array,C=10):
    y = y.reshape(100,1) * 1.
    m,n = X.shape
    Xy = y.reshape(-1,1 ) * X
    H = Xy@Xy.T
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))
    #G = cvxopt_matrix(-np.eye(m))
    G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
    #h = cvxopt_matrix(np.zeros(m))
    h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))
    solver = cvxopt.solvers.qp(P,q,G,h,A,b,)
    alphas = np.array(solver['x'])
    w = ((y * alphas).T @ X).reshape(-1,1)
    S = (alphas > 1e-4).flatten()
    SS = (alphas > 9).flatten()
    b = sum(y[S] - np.dot(X[S], w))/len(y[S])
    #b = sum([yi - for yi in y])/len(y)
    print(f"alphas not null {len(alphas[S])}")
    print('Alphas = ',alphas)
    print('w = ', w.flatten())
    print(f'b = {b}')
    return w,b,SS

y = y_classifier(y)
w,b,S = SVM_normal(X,y)
#print_x(X,y,w,b,S)
```
![](https://i.imgur.com/8VeKwtH.png)
The blue line is the decision boundary. the crossed Dots are SupportVectors or are in the Error Magrin. It's difficult to differentiate between the two because the $\alpha_i > 0$ are close to the constant $C$. 


## Task 3: Gaussian Processes

The implementation is more or less a direct implementation of the slides (lecture 13):

- RBF kernel: slide 11
- GP: slides 25 and 26

```python=
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm,inv

noise = 0.005
interval = (0,2*np.pi)
steps = int((interval[1]-interval[0])/0.005)
x_steps = np.linspace(interval[0],interval[1],steps)

#INITAL
x_lst = [x_steps[steps//2]]
y_lst = [function(x_lst[0])]
#
mu = [0]
SIG1 = np.array([[kernel(x,y) for y in x_lst]for x in x_lst]).reshape(len(x_lst),len(x_lst)) + noise*np.eye(len(x_lst))
c = np.array([1+noise]).reshape(1,1)
#FIST
k = lambda y: np.array([kernel(y,x) for x in x_lst]).reshape(-1,1)
#mu_f = lambda x: np.array(mu) + k(x).T@inv(SIG1)@np.array(y_lst).reshape(-1,1)
# assumtion mu evertime 0; should be wrong because we need to update maybe just mu_new = 0
# or should be calced evertime (y - mu)
mu_f = lambda x: k(x).T@inv(SIG1)@np.array(y_lst).reshape(-1,1)
cov_f = lambda x: c - k(x).T@inv(SIG1)@k(x)
i = 0
while i < 16:
    print(f"mu {mu} \n x {x_lst} \n y {y_lst}")
    cov_lst = [norm(cov_f(x).reshape(-1)) for x in x_steps]
    new_x = x_steps[np.argmax(cov_lst)]
    new_y = function(new_x)
    new_mu = mu_f(x_lst[-1])[0,0]
    x_lst.append(new_x)
    y_lst.append(new_y)
    mu.append(new_mu)
    SIG1 = np.array([[kernel(x,y) for y in x_lst]for x in x_lst]).reshape(len(x_lst),len(x_lst)) + noise*np.eye(len(x_lst))
    #if i in [1,2,5,10,15]:
    #    plot_me(True)
    i += 1

```

Below is the plot for iterations 1, 2, 5, 10, and 15 (note: startig count from 0). The orange line is the true function, the blue line is the mean, and the red background represents two times the standard deviation.

We have not used markers for the new points. We believe that it's easy to see which points are new since the last iteration in which we plotted.

We can see that our model becomes more confident around the areas in which new samples were added (similar to the Bayesian regression in the last exercise).

![](https://i.imgur.com/s3tyzhX.png)
