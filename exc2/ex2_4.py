import numpy as np
import matplotlib.pyplot as plt


def main():
    train_file = open("dataSets/nonParamTrain.txt")
    train_data = np.array(train_file.readlines()).astype(np.float)

    test_file = open("dataSets/nonParamTest.txt")
    test_data = np.array(test_file.readlines()).astype(np.float)

    print("== Train Data =======")
    histogram(train_data)
    kde(train_data)
    knn(train_data)

    # d) TODO: table
    print("== Test Data =======")
    kde(test_data)


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

    def V(h):
        d = 1
        return np.sqrt(2 * np.pi * h * h) ** d

    N = len(data)
    x_vals = np.linspace(-4, 8, N)

    kde_1 = [gauss_kernel(x, 0.03, data) / (N * V(0.03)) for x in x_vals]
    kde_2 = [gauss_kernel(x, 0.2, data) / (N * V(0.2)) for x in x_vals]
    kde_3 = [gauss_kernel(x, 0.8, data) / (N * V(0.8)) for x in x_vals]

    print("L(0.03) = ", np.sum(np.log(list(filter(lambda x: x != 0.0, kde_1)))))
    print("L(0.2) = ", np.sum(np.log(list(filter(lambda x: x != 0.0, kde_2)))))
    print("L(0.8) = ", np.sum(np.log(list(filter(lambda x: x != 0.0, kde_3)))))

    plt.plot(x_vals, kde_1)
    plt.plot(x_vals, kde_2)
    plt.plot(x_vals, kde_3)
    plt.legend(["$\\sigma = 0.03$", "$\\sigma = 0.2$", "$\\sigma = 0.8$"])

    plt.title("Gaussian KDE")
    plt.xlabel("x")
    plt.ylabel("$p(x)$")
    plt.show()


def gauss_kernel(x, h, data):
    d = 1  # TODO what is d?
    accum = 0.0
    for x_i in data:
        # left = (1 / (np.sqrt(2 * np.pi * h * h) ** d))
        left = 1
        right = np.exp(-(np.linalg.norm(x - x_i) ** 2) / (2 * h * h))
        accum += left * right
    return accum


def knn(data):
    print("\t kNN")

    N = len(data)
    x_vals = np.linspace(-4, 8, N)

    knn_1 = [knn_kernel(x, 2, data) for x in x_vals]
    knn_2 = [knn_kernel(x, 8, data) for x in x_vals]
    knn_3 = [knn_kernel(x, 35, data) for x in x_vals]

    print("L(2) = ", np.sum(np.log(list(filter(lambda x: x != 0.0, knn_1)))))
    print("L(8) = ", np.sum(np.log(list(filter(lambda x: x != 0.0, knn_2)))))
    print("L(35) = ", np.sum(np.log(list(filter(lambda x: x != 0.0, knn_3)))))

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
    neighbors = np.sort(np.array([np.abs(x - x_i) for x_i in data]))[:K]    # Contains line lengths
    V = np.max(neighbors) - np.min(neighbors)   # TODO: this is wrong
    V = neighbors   # A bit better but not sure if right
    return K / (N * 2 * V[K - 1])


if __name__ == "__main__":
    main()
