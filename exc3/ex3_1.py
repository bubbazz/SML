import numpy as np
import matplotlib.pyplot as plt


# TODO: in general, use inv or inv?
def main():
    train_data = np.loadtxt("dataSets/lin_reg_train.txt")
    test_data = np.loadtxt("dataSets/lin_reg_test.txt")

    x_train = train_data[:, 0]
    y_train = train_data[:, 1]
    x_test = test_data[:, 0]
    y_test = test_data[:, 1]

    # a) Linear Features
    # a(x_train, y_train, x_test, y_test)

    # b) Polynomial Features
    # b(x_train, y_train, x_test, y_test)

    # c) Bayesian Linear Regression
    c(x_train, y_train, x_test, y_test)

    # d) Squared Exponential Features
    d(x_train, y_train, x_test, y_test)

    # e) Cross Validation


def a(x, y, x_test, y_test):
    print("== Linear Features ========")

    # Fit the model using linear ridge regression with a ridge coefficient of 0.01
    x = x.reshape((len(x), 1))
    y = y.reshape((len(y), 1))
    X = np.hstack((x, np.ones_like(x))).T

    ridge = 0.01
    I = np.identity(X.shape[0])

    w = np.linalg.inv(X @ X.T + ridge * I) @ X @ y

    y_pred = X.T @ w

    # Plot and show error
    plt.title("Linear Features")
    plt.scatter(x, y, color="black")
    plt.plot(x, y_pred, color="blue")
    plt.show()

    print("RMSE (Train): ", rmse(y, y_pred))

    # Validate trained parameters w under test set
    x_test = x_test.reshape((len(x_test), 1))
    y_test = y_test.reshape((len(y_test), 1))
    X_test = np.hstack((x_test, np.ones_like(x_test))).T
    print("RMSE (Test): ", rmse(y_test, X_test.T @ w))


def b(x, y, x_test, y_test):
    print("\n== Polynomial Features ========")

    # Fit the model with a ridge coefficient of 0.01 for polynomials of degrees 2, 3, and 4
    x = x.reshape((len(x), 1))
    y = y.reshape((len(y), 1))
    x_test = x_test.reshape((len(x_test), 1))
    y_test = y_test.reshape((len(y_test), 1))

    for d in range(2, 5):
        X = np.hstack((x, np.ones_like(x)))
        X_test = np.hstack((x_test, np.ones_like(x_test)))
        for i in range(2, d + 1):
            X = np.hstack(((x ** i), X))
            X_test = np.hstack(((x_test ** i), X_test))
        X = X.T
        X_test = X_test.T

        ridge = 0.01
        I = np.identity(d + 1)

        w = np.linalg.inv(X @ X.T + ridge * I) @ X @ y

        y_pred = X.T @ w

        # Show error for train and test data
        print(f"d = {d}:")
        print("\tRMSE (Train): ", rmse(y, y_pred))
        print("\tRMSE (Test): ", rmse(y_test, X_test.T @ w))

        # Plot
        plt.title(f"Polynomial Features (d={d})")
        plt.scatter(x, y, color="black")
        # plt.scatter(x, y_pred, color="orange")  # Not required by the task description
        new_x, new_y = polynomial(x.flatten(), y_pred.flatten(), d)
        plt.plot(new_x, new_y, color="blue")
        plt.show()


def c(x, y, x_test, y_test):
    print("\n== Bayesian Linear Regression ========")

    # TODO: figure out relationshipt between alpha, beta, sigma, and lambda ... for now just use lambda as alpha/beta
    sigma = 0.1
    ridge = 0.01

    alpha = 0.01                # alpha = lambda
    beta = 1.0 / (sigma ** 2)   # beta^-1 is the variance; sigma is std; sigma^2 = beta^-1; beta=beta^-1=(sigma^2)^-1

    x = x.reshape((len(x), 1))
    y = y.reshape((len(y), 1))
    x_test = x_test.reshape((len(x_test), 1))
    y_test = y_test.reshape((len(y_test), 1))

    X = np.hstack((x, np.ones_like(x))).T
    X_test = np.hstack((x_test, np.ones_like(x_test))).T

    print("X = ", X.shape)

    I = np.identity(2)

    # Prior
    m0 = np.zeros_like(X)
    Lambda0 = ridge * np.identity(2)

    # Posterior
    # print(X.shape, y.shape)
    # print(Lambda0.shape, m0.shape, (X @ y).shape)
    LambdaN = np.linalg.inv(np.linalg.inv(Lambda0) + (sigma ** -2) * X @ X.T)
    # # print(LambdaN.shape, m0.shape)
    mN = LambdaN @ (np.linalg.inv(Lambda0) @ m0 + (sigma ** -2) * X @ y)

    # Predictive (what is x*?)
    pred_mean = np.array([x @ np.linalg.inv(X @ X.T + (alpha / beta) * I) @ X @ y for x in X.T])
    pred_var = np.array([(1 / beta) + x.T @ np.linalg.inv(beta * X @ X.T + alpha * I) @ x for x in X.T])

    pred_mean_test = np.array([x @ np.linalg.inv(X @ X.T + (alpha / beta) * I) @ X @ y for x in X_test.T])
    pred_var_test = np.array([(1 / beta) + x.T @ np.linalg.inv(beta * X @ X.T + alpha * I) @ x for x in X_test.T])
    print(pred_mean.shape, pred_var.shape)

    print(X_test.shape, y_test.shape, pred_mean_test.shape)
    print("RMSE (Train): ", rmse(y, pred_mean))
    print("RMSE (Test): ", rmse(y_test, pred_mean_test))

    # Plot
    x_tmp, pred_mean = polynomial(x.flatten(), pred_mean.flatten(), 1)
    _, pred_var = polynomial(x.flatten(), pred_var.flatten(), 1)

    x_tmp, pred_mean_test = polynomial(x_test.flatten(), pred_mean_test.flatten(), 1)
    _, pred_var_test = polynomial(x_test.flatten(), pred_var_test.flatten(), 1)

    plt.plot(x_tmp, pred_mean, color="blue")

    std_dev = np.sqrt(pred_var_test)
    pred_mean = pred_mean_test.flatten()
    plt.fill_between(
        x=x_tmp.flatten(),
        y1=pred_mean - (1 * std_dev),
        y2=pred_mean + (1 * std_dev),
        alpha=0.8,
        color="royalblue")
    plt.fill_between(
        x=x_tmp.flatten(),
        y1=pred_mean - (2 * std_dev),
        y2=pred_mean + (2 * std_dev),
        alpha=0.5,
        color="cornflowerblue")
    plt.fill_between(
        x=x_tmp.flatten(),
        y1=pred_mean - (3 * std_dev),
        y2=pred_mean + (3 * std_dev),
        alpha=0.25,
        color="lightsteelblue")

    plt.scatter(x, y, color="black", label="Train")
    plt.scatter(x_test, y_test, color="red", label="Test")
    plt.legend()
    plt.show()


# Copy pasted (c)
def d(x, y, x_test, y_test):
    print("\n== Squared Exponential Features ========")

    sigma = 0.1
    ridge = 0.01

    n = len(x)
    n_test = len(x_test)
    k = 20
    beta = 10
    alpha = np.array([j * 0.1 - 1 for j in range(k)])
    F = np.empty((n, k), dtype=float)
    F_test = np.empty((n_test, k), dtype=float)
    I = np.identity(k)

    x = x.reshape((len(x), 1))
    y = y.reshape((len(y), 1))
    x_test = x_test.reshape((len(x_test), 1))
    y_test = y_test.reshape((len(y_test), 1))

    for i in range(n):
        for j in range(k):
            F[i][j] = np.exp(-0.5 * beta * (x[i] - alpha[j]) ** 2)

    for i in range(n_test):
        for j in range(k):
            F_test[i][j] = np.exp(-0.5 * beta * (x_test[i] - alpha[j]) ** 2)

    X = F.T
    X_test = F_test.T

    # Prior
    m0 = np.zeros_like(X)
    Lambda0 = ridge * np.identity(k)

    # Posterior
    LambdaN = np.linalg.inv(np.linalg.inv(Lambda0) + (sigma ** -2) * X @ X.T)
    mN = LambdaN @ (np.linalg.inv(Lambda0) @ m0 + (sigma ** -2) * X @ y)

    # Predictive
    pred_mean = np.array([x @ np.linalg.inv(X @ X.T + (alpha / beta) * I) @ X @ y for x in X.T])
    pred_var = np.array([(1 / beta) + x.T @ np.linalg.inv(beta * X @ X.T + alpha * I) @ x for x in X.T])

    pred_mean_test = np.array([x @ np.linalg.inv(X @ X.T + (alpha / beta) * I) @ X @ y for x in X_test.T])
    pred_var_test = np.array([(1 / beta) + x.T @ np.linalg.inv(beta * X @ X.T + alpha * I) @ x for x in X_test.T])

    print("RMSE (Train): ", rmse(y, pred_mean))
    print("RMSE (Test): ", rmse(y_test, pred_mean_test))

    # Plot
    x_tmp, pred_mean = polynomial(x.flatten(), pred_mean.flatten(), k)
    _, pred_var = polynomial(x.flatten(), pred_var.flatten(), k)

    x_tmp, pred_mean_test = polynomial(x_test.flatten(), pred_mean_test.flatten(), k)
    _, pred_var_test = polynomial(x_test.flatten(), pred_var_test.flatten(), k)
    plt.plot(x_tmp, pred_mean, color="blue")

    std_dev = np.sqrt(pred_var_test)
    pred_mean = pred_mean_test.flatten()
    plt.fill_between(
        x=x_tmp.flatten(),
        y1=pred_mean - (1 * std_dev),
        y2=pred_mean + (1 * std_dev),
        alpha=0.8,
        color="royalblue")
    plt.fill_between(
        x=x_tmp.flatten(),
        y1=pred_mean - (2 * std_dev),
        y2=pred_mean + (2 * std_dev),
        alpha=0.5,
        color="cornflowerblue")
    plt.fill_between(
        x=x_tmp.flatten(),
        y1=pred_mean - (3 * std_dev),
        y2=pred_mean + (3 * std_dev),
        alpha=0.25,
        color="lightsteelblue")

    plt.scatter(x, y, color="black", label="Train")
    plt.scatter(x_test, y_test, color="red", label="Test")
    plt.legend()
    plt.show()


# TODO: I don't think this is needed
def gaussian(x, mean, cov):
    print(x.shape, mean.shape, cov.shape, len(x))
    k = len(x)
    norm = 1.0 / np.sqrt(((2 * np.pi) ** k) * np.linalg.det(cov))
    gauss = np.exp(-0.5 * (x - mean).T @ cov @ (x - mean))
    return norm * gauss


def rmse(y, y_pred):
    return np.sqrt(np.mean((y - y_pred) ** 2))


# See https://www.kite.com/python/answers/how-to-plot-a-polynomial-fit-from-an-array-of-points-using-numpy-and-matplotlib-in-python
def polynomial(x, y, d):
    coefficients = np.polyfit(x, y, d)
    poly = np.poly1d(coefficients)
    new_x = np.linspace(np.min(x), np.max(x))
    new_y = poly(new_x)
    return new_x, new_y


if __name__ == "__main__":
    main()
