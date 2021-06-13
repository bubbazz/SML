import os
from matplotlib import pyplot as plt
import numpy as np
import random
from scipy.stats import multivariate_normal

# init
random.seed(38)

N = 3
data = np.loadtxt(os.path.abspath("dataSets/gmm.txt"))
means = [(random.uniform(-8, 8), random.uniform(-8, 8)) for i in range(N)]
covs = [np.identity(2) for i in range(N)]
# pi -> sum pi == 1
mixing_coefficients = [1/N for i in range(3)]
# x, y = np.random.multivariate_normal(means[0], np.identity(2), 5000).T
t = [1, 3, 5, 10, 30]
# see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
colors_transparent = ['#1f77b422', '#2ca02c22', '#d6272822']
colors = ['#1f77b4ff', '#2ca02cff', '#d62728ff']


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


# see https://stackoverflow.com/a/48409811/4303296
def ellipse(mean, cov, color):
    eigenvals, eigenvecs = np.linalg.eig(cov)

    x, y = mean
    w, h = eigenvals
    t = np.linspace(0, 2 * np.pi, 100)

    rot = eigenvecs
    ell = np.array([w * np.cos(t), h * np.sin(t)])
    ell_rot = np.zeros((2, ell.shape[1]))
    for i in range(ell.shape[1]):
        ell_rot[:, i] = np.dot(rot, ell[:, i])

    plt.plot(x + ell_rot[0, :], y + ell_rot[1, :], color=color)


def nearest(means, data):
    N = len(means)
    points = []
    for i in range(N * 2):
        points.append([])

    for p in data:
        min = 9999
        min_idx = 0
        for i in range(N):
            norm = np.linalg.norm(np.array(means[i]) - p)
            if norm < min:
                min = norm
                min_idx = i
        points[min_idx * 2].append(p[0])
        points[min_idx * 2 + 1].append(p[1])

    return points


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
        if i in t:
            plt.title(f"t = {i}")
            points = nearest(means, data)
            for j in range(len(means)):
                ellipse(means[j], covs[j], colors[j])
                plt.scatter(points[j * 2], points[j * 2 + 1], c=colors_transparent[j])
                plt.scatter(means[j][0],
                            means[j][1], marker="x", c=colors[j])
            plt.show()
    return


circle_x = np.arange(-1, 1, 0.01)
cirlce_y = [np.sqrt(1-x**2) for x in circle_x]
# [COV_i*(x,y)+mu_i]
#plt.plot(circle_x, cirlce_y)
EMAlgo(means, covs, mixing_coefficients, t[-1] + 1)
plt.show()
