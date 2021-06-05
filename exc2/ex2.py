import numpy as np
import matplotlib.pyplot as pl
from numpy import linalg


def rosenberg(x: np.array):
    return sum([100*(xii - xi**2)**2+(xi - 1)**2 for xi, xii in zip(x[:-1], x[1:])])


def rosenberg_devirate(x, i):
    if i == len(x)-1:
        return 200*(x[-1] - x[-2]**2)
    if i == 0:
        return 2*(x[0]-1) - 400*(x[1] - x[0]**2)*x[0]
    return 2*(x[i]-1) - 400*(x[i+1] - x[i]**2)*x[i] + 200*(x[i] - x[i-1])


def gradient_descent(x: np.array, alpha):
    gradient = np.array([rosenberg_devirate(x, i) for i in range(len(x))])
    x_hat = x - alpha*gradient
    return x_hat


np.set_printoptions(formatter={'all': lambda x: str(x)})
x = np.array([-1, -1])
lst = [x]
for i in range(10000):
    lst.append(gradient_descent(lst[-1], 0.00255))
lst = np.array(lst)
error = np.array([np.linalg.norm(np.array([1, 1]) - e)for e in lst])
error = np.column_stack((error, np.array(range(len(error)))))
#pl.scatter(lst[:, 0], lst[:, 1])
pl.scatter(error[:, 1], error[:, 0])
pl.xlabel("$steps$")
pl.ylabel("$|(1,1)^T - grad(steps)|_2$")
pl.title("the Distance for the error ")
pl.show()
