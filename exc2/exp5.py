import os
from matplotlib import pyplot as plt
import numpy as np
import random
from scipy.stats import multivariate_normal

# init
N = 3
data = np.loadtxt(os.path.abspath("dataSets/gmm.txt"))
means = [(random.uniform(-8, 8), random.uniform(-8, 8)) for i in range(N)]
covs = [np.identity(2) for i in range(N)]
# pi -> sum pi == 1
mixing_coefficients = [1/N for i in range(3)]
# x, y = np.random.multivariate_normal(means[0], np.identity(2), 5000).T
plt.scatter(data[:, 0], data[:, 1])


def Expectation(means, covs, mix_coefs, X):
    ''' Compute the posterior distribution for each mixture component and for all data points'''
    # 1 : x_1 : [pi_1*N(x_1|mu_1,cov_1),pi_2*N(x_1|mu_2,cov_2),pi_3*N(x_1|mu_3,cov_3)]
    # 2 : x_2 : [pi_1*N(x_2|mu_1,cov_1),pi_2*N(x_2|mu_2,cov_2),pi_3*N(x_2|mu_3,cov_3)]
    all_we_need = [[mix_coef*multivariate_normal.pdf(x=x, mean=mean, cov=cov)
                    for mean, cov, mix_coef in zip(means, covs, mix_coefs)] for x in X]
    # [pi_1*N(x_1|mu_1,cov_1)/sum(pi_1*N(x_1|mu_1,cov_1),pi_2*N(x_1|mu_2,cov_2),pi_3*N(x_1|mu_3,cov_3)),
    # pi_2*N(x_1|mu_2,cov_2)/sum(pi_1*N(x_1|mu_1,cov_1),pi_2*N(x_1|mu_2,cov_2),pi_3*N(x_1|mu_3,cov_3))]...
    alphas = [[x/sum(xs) for x in xs] for xs in all_we_need]
    return alphas


def Maximization(X, alpha_list):
    Nks = [sum(np.array(alpha_list)[:, i])
           for i in range(len(alpha_list[0]))]
    new_mix_coefs = [nk/len(X) for nk in Nks]
    new_means = np.array([[alpha*x for alpha in alphas]
                          for alphas, x in zip(alpha_list, X)])
    new_means = [sum(new_means[:, i])/Nks[i]
                 for i in range(len(new_means[0]))]
    # vec*vec.T
    new_cov = np.array([[alpha*np.matrix(x-new_mean).T*np.matrix((x-new_mean)) for alpha, new_mean in zip(alphas, new_means)]
                        for alphas, x in zip(alpha_list, X)])
    new_cov = [sum(new_cov[:, i])/Nks[i]
               for i in range(len(new_cov[0]))]
    return new_means, new_cov, new_mix_coefs


def EMAlgo(means, covs, mix_coefs, steps):
    struc = {"mean": [means],
             "covs": [covs],
             "mix_coefs": [mix_coefs]}
    for i in range(steps):
        alphas = Expectation(means, covs, mixing_coefficients, data)
        means, covs, mix_coefs = Maximization(data, alphas)
        struc["mean"].append(means)
        struc["covs"].append(covs)
        struc["mix_coefs"].append(mix_coefs)
    for i in range(len(struc["mean"][0])):
        plt.scatter(np.array(struc["mean"])[:, i][:, 0],
                    np.array(struc["mean"])[:, i][:, 1], marker="x")
    return


circle_x = np.arange(-1, 1, 0.01)
cirlce_y = [np.sqrt(1-x**2) for x in circle_x]
# [COV_i*(x,y)+mu_i]
#plt.plot(circle_x, cirlce_y)
EMAlgo(means, covs, mixing_coefficients, 10)
plt.show()
