import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def main():
    np.set_printoptions(suppress=True)

    data = np.loadtxt("dataSets/iris.txt", delimiter=",")
    iris_data, labels = data[:, 0:4], data[:, 4]

    label_names = ["Setosa", "Versicolour", "Virginica"]
    colors = ['#1f77b4ff', '#2ca02cff', '#d62728ff']

    plt.title("Raw Data")
    sc = plt.scatter(iris_data[:, 0], iris_data[:, 1], c=labels, cmap=ListedColormap(colors))
    plt.legend(handles=sc.legend_elements()[0], labels=label_names)
    plt.show()

    # Get normalized data and mean
    iris_data_normalized, mean, std = normalize(iris_data)
    plt.title("Normalized Data")
    sc = plt.scatter(iris_data_normalized[:, 0], iris_data_normalized[:, 1], c=labels, cmap=ListedColormap(colors))
    plt.legend(handles=sc.legend_elements()[0], labels=label_names)
    plt.show()

    B, cumul_var = pca(iris_data_normalized)
    plt.title("Cumulative Variance")
    plt.xlabel("Components")
    plt.ylabel("Explained Variance")
    plt.plot([1, 2, 3, 4], cumul_var)
    plt.xticks([1, 2, 3, 4])
    plt.show()

    B_tmp = B   # Original B needed later
    B = basis(B, cumul_var, 0.95)
    print(f"Need {B.shape[1]} components to explain at least 95% of the dataset variance.")

    proj = B @ B.T @ iris_data_normalized
    plt.title("PCA (0.95)")
    sc = plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap=ListedColormap(colors))
    plt.legend(handles=sc.legend_elements()[0], labels=label_names)
    plt.show()

    for i in range(1, 5):
        B = B_tmp[:, :i]
        proj = B @ B.T @ iris_data_normalized
        print(f"NRMSE (M = {i}): ", nrmse(iris_data_normalized, proj, mean))
        plt.title(f"PCA (M = {i})")
        sc = plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap=ListedColormap(colors))
        plt.legend(handles=sc.legend_elements()[0], labels=label_names)
        plt.show()


def normalize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data = (data - mean) / std
    return data, mean, std


def pca(data):
    N = data.shape[0]
    u, s, _ = np.linalg.svd(data)
    eigvals = (1.0 / N) * s * s
    cumul_var = np.cumsum((eigvals / np.sum(eigvals)))
    return u, cumul_var


# Assumes t in [0, 1]
def basis(u, cumul_var, t=0.95):
    total_var = cumul_var[-1]

    D = 0
    for i, var in enumerate(cumul_var):
        if var >= t * total_var:
            D = i + 1
            break

    return u[:, :D]


def nrmse(x, x_tilde, mean):
    return (1.0 / mean) * np.sqrt(np.linalg.norm(x - x_tilde, axis=0) ** 2)


if __name__ == "__main__":
    main()
