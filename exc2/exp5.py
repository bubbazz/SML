import os
from matplotlib import pyplot as plt
import numpy as np
import random
from scipy.stats import multivariate_normal

# init
data = np.loadtxt(os.path.abspath("dataSets/gmm.txt"))
mean = [(random.uniform(-3, 8), random.uniform(-3, 8)) for i in range(3)]
print([(mean[0], d, multivariate_normal.pdf(d, mean[0])) for d in data[0:10]])


def Expectation(mean, pdf):
    return []


def Maximization():
    return


def EMAlgo(means, steps):
    struc = [(mean, multivariate_normal.pdf(mean))for mean in means]
    for i in range(steps):
        for j in range(len(struc)):
            probabiltys = Exception(struc[j][0], struc[j][1])
