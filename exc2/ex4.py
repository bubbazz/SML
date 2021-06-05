import matplotlib.pyplot as plt
import numpy as np
import os
import math


train_data = np.loadtxt(os.path.abspath("dataSets/nonParamTrain.txt"))
test_data = np.loadtxt(os.path.abspath("dataSets/nonParamTest.txt"))
# range(min(data), max(data) + binwidth, binwidth)


def excA(data):
    binlen = [0.2, 0.5, 2]
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    for i in range(3):
        binssize = int((max(data) - min(data)) / binlen[i])
        ax[i].hist(data, bins=binssize)
    plt.show()


# i think it should be used the middle binlength because u can "perfectly" see 2 gaußians with
# mean also u can guess the varianz (Standartabweiung)

def gauß_kernel(x, sigma):
    return math.exp(-np.linalg.norm(x)**2/(2*sigma**2))/math.sqrt(2*math.pi*sigma**2)


def probabilty_kernel(X, sigma):
    # d = X[0] or 1 ; len(X) = N ; K()
    d = len(X[0]) if X[0] is list else 1
    const = (len(X)*math.sqrt(2*math.pi*sigma**2)**d)
    # norming = sum([gauß_kernel(x, sigma) for x in X])
    return lambda x: sum([gauß_kernel(x-xn, sigma) for xn in X])/(const)


def excB(data):
    f = [probabilty_kernel(data, 0.03), probabilty_kernel(
        data, 0.8), probabilty_kernel(data, 0.2)]
    for i in range(len(f)):
        loglikelihood(data, f[i])
    # not normed but dont need to
    for i in range(len(f)):
        plt.plot(np.arange(-4.0, 8.2, 0.2),
                 [f[i](x) for x in np.arange(-4.0, 8.2, 0.2)])
    plt.show()

# bullshit maybeusefull for later so i commit it


def KNN_old(x, X, k):
    sortedX = sorted(X)
    Volumen = []
    for i in range(len(X)):
        start = 0 if i-k-1 < 0 else i-k-1
        end = len(X)-1 if i+k >= len(X) else i+k+1
        Volumen.append(sorted([np.linalg.norm(sortedX[i]-x)
                               for x in sortedX[start:end]])[k])


def KNN(X, k):
    '''k = K;  N = len(x) ; V = sorted([np.linalg.norm(x-el) for el in X])[k]
        p(x) = K/(NV(x))'''
    return lambda x: k/(len(X)*sorted([np.linalg.norm(x-el) for el in X])[k])


def excC(data):
    knn = [KNN(data, 2), KNN(data, 8), KNN(data, 35)]
    for i in range(3):
        plt.plot(np.arange(-4.0, 8.2, 0.2),
                 [knn[i](x) for x in np.arange(-4.0, 8.2, 0.2)])
    plt.show()


def loglikelihood(X, P):
    # TODO L(theta) oc P(x|theta)
    return lambda teta:  sum([math.log(P(xx))for xx in X])

# TODO dx loglikelihood == 0


def newton(f, x, epsilon):
    while epsilon:
        x -= f(x)/(f(x)-f(x+epsilon))


def excD(f, data):
    loglike = loglikelihood(data, f)
    x0 = 0
    theta = newton(loglike, x0)
    return theta


# excA(train_data)
# excB(train_data)
excC(train_data)
