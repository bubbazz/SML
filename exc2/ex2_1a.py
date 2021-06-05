import numpy as np
from matplotlib import pyplot as plt

# Global variables
n = 20
num_steps = 10000
t_fix = 0.001


def main():
    # Turn off scientific notation
    np.set_printoptions(formatter={'all': lambda s: str(s)})

    # Generate a random n-dim vector and evaluate it
    # Multiply with 1.0 to convert to float
    # x = np.random.randint(0, 5, (n, 1)) * 1.0     # Fixed t cannot deal with arbitrary ranges
    x = np.zeros(n)                                 # Works well for both
    x_fix = x

    x_prime = np.empty_like(x, dtype=float)
    x_prime_fix = np.empty_like(x, dtype=float)

    learning_rate = [rosenbrock(x)]
    learning_rate_fix = [rosenbrock(x_fix)]

    # print("x = ", x)
    print("Before: ", rosenbrock(x))

    for step in range(num_steps):
        if step % 1000 == 0:
            print("Step: ", step)

        # Calculate f'(x_i)
        for i in range(n):
            x_prime[i] = rosenbrock_derivative(x, i)
            x_prime_fix[i] = rosenbrock_derivative(x_fix, i)

        # Fixed step size
        t = t_fix
        x_fix = x_fix - t * x_prime_fix

        # Adaptive step size
        t = backtrack_line_search(x, x_prime)
        x = x - t * x_prime

        # Append new results so we can see how the function evolved over time
        learning_rate.append(rosenbrock(x))
        learning_rate_fix.append(rosenbrock(x_fix))

    # print(x)
    print("After: ", rosenbrock(x))

    plot_learning_rate(learning_rate, learning_rate_fix)


# x is a vector
# sum_0^(n-1): 100 * (x_i+1 - x_i ** 2) ** 2 + (x_i - 1) ** 2
def rosenbrock(x):
    accum = 0
    for i in range(n - 1):
        accum += 100 * ((x[i + 1] - x[i] ** 2) ** 2) + ((x[i] - 1) ** 2)
    return accum


def rosenbrock_derivative(x, i):
    if i == 0:
        return 2 * ((200 * x[i] ** 3) - (200 * x[i] * x[i + 1]) + x[i] - 1)
    elif i == n - 1:
        return 200 * (x[i] - x[i - 1] ** 2)
    else:
        return (-200 * x[i - 1] ** 2) + (400 * x[i] ** 3) + (x[i] * (202 - 400 * x[i + 1])) - 2


# Does a backtracking line search to determine a good approximation for the current step size
# see https://www.cs.cmu.edu/~ggordon/10725-F12/scribes/10725_Lecture5.pdf
# see https://people.cs.umass.edu/~barna/2015-BigData/conv2.pdf
def backtrack_line_search(x, x_prime):
    # Alpha should be in (0, 0.5), beta in (0, 1)
    alpha = 0.25
    beta = 0.5

    # Cache these values to avoid recalculation at every iteration
    f_x = rosenbrock(x)
    dir_norm_squared = np.linalg.norm(x_prime) ** 2

    t = 1
    while rosenbrock(x - t * x_prime) > (f_x - alpha * t * dir_norm_squared):
        t *= beta

    return t


def plot_learning_rate(learning_rate, learning_rate_fix):
    plt.title(f"Learning Rates (Adaptive w/ Backtracking vs. Fixed)\n$x_0 = 0$")
    plt.xlabel("Step")
    plt.ylabel("f(x)")
    plt.plot(learning_rate)
    plt.plot(learning_rate_fix)
    plt.legend(["$t = backtrack(x, \\Delta f(x))$", f"$t = {t_fix}$"])
    # plt.yscale("log")
    plt.savefig("Ex2_Task1_GD_Plot.png")
    plt.show()


if __name__ == "__main__":
    main()
