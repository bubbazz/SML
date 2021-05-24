import numpy as np
from matplotlib import pyplot as plt
import math


def iterate(n):
    markov_chain_matrix = np.array([[0.42, 0.026], [0.58, 0.974]])
    s0 = [1, 0]
    slist = [s0]
    for i in range(n):
        print(i, s0)
        s0 = markov_chain_matrix.dot(s0)
        slist.append(s0)
    print(n, s0)
    return slist


def plotGenerations(slist, n):
    # plt.style.use('seaborn')
    plt.plot(range(n + 1), [x[0] for x in slist])
    plt.plot(range(n + 1), [x[1] for x in slist])
    if n == 18:
        plt.xticks(np.arange(0, n + 1, 1))
    plt.title(f"{n} days")
    plt.xlabel("Days")
    plt.ylabel("%")
    plt.legend(['$m$', '$\\tilde{m}$'])
    plt.show()


# n = 18
# slist = iterate(n)
# plotGenerations(slist, n)

# n = 1000
# slist = iterate(n)
# plotGenerations(slist, n)

def eigenvalues():
    markov_chain_matrix = np.array([[0.42, 0.026], [0.58, 0.974]])
    print(markov_chain_matrix, markov_chain_matrix.T)
    lamb, vec = np.linalg.eig(markov_chain_matrix)
    print("eigenvalue_1 :" + str(lamb[1]) +
          "\neigenvector_1 : " + str(vec[:, 1]))
    # analitically calculated values
    wolfram_mat = [13/290, 1]
    stochastical_norm = sum([x for x in wolfram_mat])
    wolfram_mat_normed = (1/stochastical_norm)*np.array(wolfram_mat)
    print("normed calc:" + str(wolfram_mat_normed))
    markov_chain_matrix = markov_chain_matrix.T
    n = 10000
    x = np.linalg.matrix_power(markov_chain_matrix, n)
    print(n, x)


eigenvalues()
