import numpy as np
from matplotlib import pyplot as plt
import os

from numpy.ma import exp

data1 = np.loadtxt(os.path.abspath("dataSets/densEst1.txt"))
data2 = np.loadtxt(os.path.abspath("dataSets/densEst2.txt"))
plt.scatter(data1[:, 0], data1[:, 1])
plt.scatter(data2[:, 0], data2[:, 1])

# a
all_data = len(data1)+len(data2)
prior_C1 = len(data1)/all_data
prior_C2 = len(data2)/all_data
# b
# the maximumlikelihood is an unbiased estimator for the mean
# but for the variance it is biased
mean_1 = sum(data1)/len(data1)
mean_2 = sum(data2)/len(data2)
print(np.shape(mean_1))
print()


def cov(biased: bool, mean, X):
    length = len(X) if biased else len(X)-1
    return sum([np.matrix([x-mean]).T*np.matrix([x-mean])
                for x in X])/length


cov_1_biased = cov(True, mean_1, data1)
print(np.shape(cov_1_biased))
cov_1_unbiased = cov(False, mean_1, data1)
cov_2_biased = cov(True, mean_2, data2)
cov_2_unbiased = cov(False, mean_2, data2)
# c


def multivariate_gauß(x, mean, cov):
    return np.exp(-np.matrix([x-mean]).T*cov*np.matrix([x-mean])) /\
        np.sqrt((2)**(np.shape(x)[0])*np.linalg.det(cov))


xlist = np.linspace(-10.0, 10.0, 100)
ylist = np.linspace(-8.0, 8.0, 100)
X, Y = np.meshgrid(xlist, ylist)
# TODO
# Z = --- np.exp(X+Y)
fig, ax = plt.subplots(1, 1)
cp = ax.contourf(X, Y, Z)
fig.colorbar(cp)  # Add a colorbar to a plot
