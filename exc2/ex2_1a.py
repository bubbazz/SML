import numpy as np
from matplotlib import pyplot as plt

# Global variables
n = 20
num_steps = 10000
alpha = 0.00001


def main():
    # Turn off scientific notation
    np.set_printoptions(formatter={'all': lambda x: str(x)})

    x = np.random.randint(0, 100, (n, 1)) * 1.0
    print("x = ", x)
    rosenbrock(x)

    alpha = 0.00001
    sol = x
    x_prime = np.empty_like(x, dtype=float)
    for step in range(num_steps):
        # Calculate f'(x_i)
        for i in range(n):
            x_prime[i] = derivative(sol, i)
        # Step: x_i+1 = x_i - alpha * f'(x_i)
        # sol = sol - alpha * x_prime

        # TODO Adaptive step:
        norm_sol = np.linalg.norm(sol)
        sol_tmp = sol - alpha * x_prime
        while np.linalg.norm(sol_tmp) > norm_sol or np.linalg.norm(sol_tmp) * 2 < norm_sol:
            # Value increased, step size was too large:
            if np.linalg.norm(sol_tmp) > norm_sol:
                alpha = alpha / 10.0
                sol_tmp = sol - alpha * x_prime
            elif np.linalg.norm(sol_tmp) * 2 < norm_sol:
                alpha = alpha * 10.0
                sol_tmp = sol - alpha * x_prime
                break
        sol = sol_tmp


    print(sol)
    rosenbrock(sol)


# x is a vector
# sum_0^(n-1): 100 * (x_i+1 - x_i ** 2) ** 2 + (x_i - 1) ** 2
def rosenbrock(x):
    accum = 0
    for i in range(n - 1):
        accum += 100 * ((x[i + 1] - x[i] ** 2) ** 2) + ((x[i] - 1) ** 2)
    print("sum = ", accum)
    return accum


def derivative(x, i):
    if i == 0:
        # return (400 * x1 ** 3) - (400 * x1 * x2) + (2 * x1) - 2
        return 2 * ((200 * x[i] ** 3) - (200 * x[i] * x[i + 1]) + x[i] - 1)
    elif i == n - 1:
        return 200 * (x[i] - x[i - 1] ** 2)
    else:
        return (-200 * x[i - 1] ** 2) + (400 * x[i] ** 3) + (x[i] * (202 - 400 * x[i + 1])) - 2


if __name__ == "__main__":
    main()
