import numpy as np
import matplotlib.pyplot as plt


def main():
    c1_data = np.loadtxt("dataSets/densEst1.txt", dtype=float)
    c2_data = np.loadtxt("dataSets/densEst2.txt", dtype=float)

    # Prior probabilities
    num_c1 = len(c1_data)
    num_c2 = len(c2_data)
    num_total = num_c1 + num_c2

    prior_c1 = (num_c1 / num_total)
    prior_c2 = (num_c2 / num_total)
    print("P(C1) = ", prior_c1)
    print("P(C2) = ", prior_c2)

    # Gaussian distribution
    delta = 0.25
    x = np.arange(-10, 11, delta)
    y = np.arange(-6, 9, delta)
    xx, yy = np.meshgrid(x, y)

    gauss_c1 = gaussian(xx, yy, c1_data)
    plt.contour(xx, yy, gauss_c1)

    gauss_c2 = gaussian(xx, yy, c2_data)
    plt.contour(xx, yy, gauss_c2)

    plt.scatter(c1_data[:, 0], c1_data[:, 1])
    plt.scatter(c2_data[:, 0], c2_data[:, 1])

    plt.title("Data with Gaussian Distribution")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(["C1", "C2"])
    plt.show()

    # Posterior probabilities
    likelihoods_c1 = gauss_c1
    likelihoods_c2 = gauss_c2
    normalization = likelihoods_c1 * prior_c1 + likelihoods_c2 * prior_c2
    posterior_c1 = (likelihoods_c1 * prior_c1) / normalization
    posterior_c2 = (likelihoods_c2 * prior_c2) / normalization

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, (likelihoods_c1 * prior_c1))
    ax.plot_surface(xx, yy, (likelihoods_c2 * prior_c2))
    plt.title("Likelihood $\\times$ Prior")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, posterior_c1)
    ax.plot_surface(xx, yy, posterior_c2)
    plt.title("Posterior")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

    plt.contour(xx, yy, (posterior_c1 - posterior_c2), levels=0)
    plt.scatter(c1_data[:, 0], c1_data[:, 1])
    plt.scatter(c2_data[:, 0], c2_data[:, 1])
    plt.title("Decision Boundary")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(["C1", "C2"])
    plt.show()


def mean(x):
    x_bar = 0.0
    for x_n in x:
        x_bar += x_n
    return (1.0 / len(x)) * x_bar


def biased_cov(x):
    cov = np.zeros((x.shape[1], x.shape[1]))
    mu = mean(x)
    for x_n in x:
        v = np.array((x_n - mu)).reshape((2, 1))
        cov += v @ v.T
    return (1.0 / len(x)) * cov


def unbiased_cov(x):
    cov = np.zeros((x.shape[1], x.shape[1]))
    mu = mean(x)
    for x_n in x:
        v = np.array((x_n - mu)).reshape((2, 1))
        cov += v @ v.T
    return (1.0 / (len(x) - 1)) * cov


def gaussian(x, y, data):
    gauss = np.zeros_like(x)
    print(gauss.shape)

    mu = mean(data)
    cov = unbiased_cov(data)
    cov_det = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)

    norm = (1.0 / np.sqrt((2 * np.pi) ** 2 * cov_det))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            y_i = y[i][j]
            x_j = x[i][j]
            v = np.array([y_i, x_j])
            diff = np.array(v - mu).reshape(2, 1)
            gauss[i][j] = norm * np.exp(-0.5 * diff.T @ cov_inv @ diff)

    return gauss


if __name__ == "__main__":
    main()