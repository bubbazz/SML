import numpy as np
import matplotlib.pyplot as plt


def main():
    train_data = np.loadtxt("dataSets/nonParamTrain.txt")
    test_data = np.loadtxt("dataSets/nonParamTest.txt")

    print("== Train Data =======")
    histogram(train_data)
    kde(train_data)
    knn(train_data)

    print("== Test Data =======")
    kde(test_data)
    knn(test_data)


def histogram(data):
    plt.subplot(311)
    plt.hist(data, bins=np.arange(data.min(), data.max(), 0.2))
    plt.title("Histograms")
    plt.ylabel("$\\Delta 0.2$")

    plt.subplot(312)
    plt.hist(data, bins=np.arange(data.min(), data.max(), 0.5))
    plt.ylabel("$\\Delta 0.5$")

    plt.subplot(313)
    plt.hist(data, bins=np.arange(data.min(), data.max(), 2.0))
    plt.ylabel("$\\Delta 2.0$")

    plt.xlabel("x")
    plt.show()


def kde(data):
    print("\t KDE")

    def gauss_norm_factor(h):
        d = 1
        return np.sqrt(2 * np.pi * h * h) ** d

    # Log-likelihoods
    N_ll = len(data)
    kde_1_ll = [gauss_kernel(x, 0.03, data) / (N_ll * gauss_norm_factor(0.03)) for x in data]
    kde_2_ll = [gauss_kernel(x, 0.2, data) / (N_ll * gauss_norm_factor(0.2)) for x in data]
    kde_3_ll = [gauss_kernel(x, 0.8, data) / (N_ll * gauss_norm_factor(0.8)) for x in data]
    print("L(0.03) = ", np.sum(np.log(list(map(lambda x: 1e-11 if x == 0 else x, kde_1_ll)))))
    print("L(0.2) = ", np.sum(np.log(list(map(lambda x: 1e-11 if x == 0 else x, kde_2_ll)))))
    print("L(0.8) = ", np.sum(np.log(list(map(lambda x: 1e-11 if x == 0 else x, kde_3_ll)))))

    N = 1000
    x_vals = np.linspace(-4, 8, N)

    kde_1 = [gauss_kernel(x, 0.03, data) / (N * gauss_norm_factor(0.03)) for x in x_vals]
    kde_2 = [gauss_kernel(x, 0.2, data) / (N * gauss_norm_factor(0.2)) for x in x_vals]
    kde_3 = [gauss_kernel(x, 0.8, data) / (N * gauss_norm_factor(0.8)) for x in x_vals]

    plt.plot(x_vals, kde_1)
    plt.plot(x_vals, kde_2)
    plt.plot(x_vals, kde_3)
    plt.legend(["$\\sigma = 0.03$", "$\\sigma = 0.2$", "$\\sigma = 0.8$"])

    plt.title("Gaussian KDE")
    plt.xlabel("x")
    plt.ylabel("$p(x)$")
    plt.show()


def gauss_kernel(x, h, data):
    accum = 0.0
    for x_i in data:
        accum += np.exp(-(np.linalg.norm(x - x_i) ** 2) / (2 * h * h))
    return accum


def knn(data):
    print("\t kNN")

    # log-likelihoods
    knn_1_ll = [knn_kernel(x, 2, data) for x in data]
    knn_2_ll = [knn_kernel(x, 8, data) for x in data]
    knn_3_ll = [knn_kernel(x, 35, data) for x in data]
    print("L(2) = ", np.sum(np.log(list(map(lambda x: 1e-11 if x == 0 else x, knn_1_ll)))))
    print("L(8) = ", np.sum(np.log(list(map(lambda x: 1e-11 if x == 0 else x, knn_2_ll)))))
    print("L(35) = ", np.sum(np.log(list(map(lambda x: 1e-11 if x == 0 else x, knn_3_ll)))))

    N = 1000
    x_vals = np.linspace(-4, 8, N)

    knn_1 = [knn_kernel(x, 2, data) for x in x_vals]
    knn_2 = [knn_kernel(x, 8, data) for x in x_vals]
    knn_3 = [knn_kernel(x, 35, data) for x in x_vals]

    plt.plot(x_vals, knn_1)
    plt.plot(x_vals, knn_2)
    plt.plot(x_vals, knn_3)
    plt.legend(["$K = 2$", "$K = 8$", "$K = 35$"])

    plt.title("kNN")
    plt.xlabel("x")
    plt.ylabel("$p(x)$")
    plt.show()


def knn_kernel(x, K, data):
    N = len(data)
    V = np.sort(np.array([np.abs(x - x_i) for x_i in data]))   # Contains line lengths
    return K / (N * 2 * V[K])


if __name__ == "__main__":
    main()
