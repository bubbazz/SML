# SML Exercise 2 (Group 38: Javor and Bekir)

## Task 1: Optimization

#### a) Numerical Optimization

![](https://i.imgur.com/WEgJ6X7.png)


Rosenbrock's function and its derivative:
```python
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
        return (-200 * x[i - 1] ** 2) + (400 * x[i] ** 3) \
        + (x[i] * (202 - 400 * x[i + 1])) - 2
```

Our loop for the gradient descent algorithm:
```python
for step in range(num_steps):
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
```

We have used both a fixed step size as well as an adaptive step size. In the above plot, we start at $\vec{x} = \vec{0}$ and use a step size of $0.001$ for the fixed version. The adaptive version uses backtracking line search which is implemented as follows:

```python
# Does a backtracking line search to determine a good approximation 
# for the current step size
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
```

Backtracking line search [1] is an approximate line search algorithm that estimates the next step size to be taken. It depends on two constants $\alpha \in (0, \tfrac{1}{2})$ and $\beta \in (0, 1)$. In our case $\alpha = \tfrac{1}{4}$ and $\beta = \tfrac{1}{2}$ gave good results that often (for arbitrary $\vec{x}_0$) were near 0 while being relatively fast.

A fixed step size is simpler to implement and is computationally less expensive. However, it requires precise tweaking as it may converge too slowly if it's set too small, or not converge at all if it's set too large.
To make matters worse, we have observed that even choosing a different starting vector $\vec{x}_0$ may require tweaking of the step size. The further away $\vec{x}_0$ was from the origin $\vec{0}$, the smaller we had to set the step size, otherwise the calculations would overflow very quickly. We could counter this by choosing a very small step size, but that affects the convergence speed such that even 10k steps may not suffice.

The adaptive step size, while computationally more costly, is able to deal with arbitrary starting points $\vec{x}_0$ and runs at reasonable speeds. Below is an additional plot for some arbitrary $\vec{x}_0$ where we have chosen a very small fixed step size of $0.00000001$. 
The adaptive version converges quickly whereas the fixed version still is far off even after 10k steps:

![](https://i.imgur.com/SuX1pE7.png)


[1] https://people.cs.umass.edu/~barna/2015-BigData/conv2.pdf

#### b) Gradient Descent Variants

##### 1)

Differences between gradient descent variants [2]:

- Batch: this is the classic approach, the one we've implemented above ($\vec{x}_{k+1} = \vec{x}_k - t\nabla f(\vec{x}_k)$)
    - It considers the whole dataset, so it is the most accurate version
    - However, it can be computationally expensive if the dataset is large
    - Furthermore, the end result highly depends on the starting point, so it may end up in a local minimum

- Stochastic: performs update for each training example and its corresponding label
    - Tends to be faster than batch GD
    - Can deal with new examples added into the model
    - Can lead to fluctuations in its objective function which allows it to escape local minima
    - Same fluctuations can lead it to overshoot the actual global minimum, but this can be mitigated by reducing the learning rate

- Mini-batch: performs update for a set of training examples and their corresponding labels
    - Basically SGD, but more accurate as it considers a set of examples
    - More stable convergence than SGD
    - Allows for trade-off: the larger the set, the more accurate it is while being slower to compute


[2] https://ruder.io/optimizing-gradient-descent/index.html#gradientdescentvariants

##### 2)

Gradient descent often exhibits a zig-zag pattern when it tries to converge using a line search algorithm, which can lead to slower convergence speeds.

Momentum can be used to improve the convergence speed. It is a term $\mu (\vec{x}_{k} - \vec{x}_{k-1})$ where $\mu \in [0, 1]$ is a weight controlling the impact of this term. When applied to gradient descent, we get $\vec{x}_{k+1} = \vec{x}_k - t_k\nabla f(\vec{x}_k) + \mu_k (\vec{x}_{k} - \vec{x}_{k-1})$.

It is especially useful when the search space has "flat" regions or when the objective function changes a lot (source: MML book).

## Task 2: Density Estimation - MLE

#### a)  Maximization Likelihood Estimate of the Exponential Distribution

Assumption we consider only positive x. because the probability function becomes zero at negative x. and thus the logrithm becomes negative infinite . therefore also the sum of the logs
\begin{align}
p(x|s) &= \frac{1}{s}exp(\frac{x}{s}) \\
L(s) &= p(X|s) = \prod_{i=1}^N p(x_i|s) = \prod_{i=1}^N \frac{1}{s}exp(\frac{x_i}{s}) =  \frac{1}{s^N}\prod_{i=1}^N exp(\frac{x_i}{s}) \\
log(L(s)) &= -N log(s)+\sum_{i=1}^N log(exp(\frac{x_i}{s})) \\
&= -N log(s) +\sum_{i=1}^N\frac{x_i}{s} \\
\end{align}
we can maximize the loglikihood with the partial derivative setting zero $\partial_s L(s) = 0$ 

\begin{align}
0 &=\partial_s L(s) = -\frac{N}{s} - \sum_{i=1}^N\frac{x_i}{s^2}\\
0 &= -\frac{sN+\sum_{i=1}^N x_i}{s^2} \\
s &= \frac{\sum_{i=1}^N x_i}{N}
\end{align}


## Task 3: Density Estimation

#### a)

```python 
num_c1 = len(c1_data)
num_c2 = len(c2_data)
num_total = num_c1 + num_c2
print("P(C1) = ", (num_c1 / num_total))
print("P(C2) = ", (num_c2 / num_total))
```

The priors are as follows:

```
P(C1) =  0.24
P(C2) =  0.76
```

#### b)

The bias can be defined as $E[\theta - \theta_*]$ where $\theta_*$ is the true parameter value. If this bias equals $0$, the estimator is said to be unbiased.

For the Gaussian we have two parameters

- the mean $\bar{\vec{x}} = \tfrac{1}{N}\sum_{n=1}^N \vec{x}_n$
- the (co-)variance $\sum = \tfrac{1}{N}\sum_{n=1}^N (\vec{x}_n - \bar{\vec{x}})(\vec{x}_n - \bar{\vec{x}})^T$

The mean is unbiased whereas the covariance is biased. We can make the covariance unbiased by replacing $N$ in the nominator with $N - 1$:

$$\sum = \tfrac{1}{N - 1}\sum_{n=1}^N (\vec{x}_n - \bar{\vec{x}})(\vec{x}_n - \bar{\vec{x}})^T$$


Sources (interestingly same chapter number for both books): 
[6.4.2 of Kevin Murphy, Machine Learning - A Probabilistic Perspective]
[6.4.2 of Marc Deisenroth, Mathematics for Machine Learning]

Our computed (unbiased) mean as well as the biased and unbiased covariances are as follows:
```
== C1 ====================
Mean:
[-0.705371   -0.81350762]

Cov (biased):
[[8.98244198 2.66170741]
 [2.66170741 3.58135631]]
 
Cov (unbiased):
[[9.02002542 2.67284426]
 [2.67284426 3.59634107]]
 
== C2 ====================
Mean:
[3.98795211 3.98714188]

Cov (biased):
[[4.17569303 0.02214194]
 [0.02214194 2.75079593]]
 
Cov (unbiased):
[[4.1811946  0.02217111]
 [0.02217111 2.75442017]]
```

They are calculated as follows:
```pyhton 
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
```

#### c)

![](https://i.imgur.com/zhtH0Ph.png)

The Gaussian is computed as follows for a grid:
```python
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
```

The plots are generated as follows:
```python 
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
```

#### d)

![](https://i.imgur.com/OjLoEzh.png)

![](https://i.imgur.com/ZExOEDx.png)

We calculated the priors in **a** and the likelihoods in **c**. The normalization is just the sum of the products of these. With these three calculating the posterior becomes easy:

```python
likelihoods_c1 = gauss_c1
likelihoods_c2 = gauss_c2
normalization = likelihoods_c1 * prior_c1 + likelihoods_c2 * prior_c2
posterior_c1 = (likelihoods_c1 * prior_c1) / normalization
posterior_c2 = (likelihoods_c2 * prior_c2) / normalization
```

The posterior plot has been generated as follows (likelihood * prior analogous):
```python
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, posterior_c1)
ax.plot_surface(xx, yy, posterior_c2)
plt.title("Posterior")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
```

The decision boundary is where $p(C_1|x) = p(C_2|x)$. So we can just look for the places where $p(C_1|x) - p(C_2|x) = 0$:
```python
plt.contour(xx, yy, (posterior_c1 - posterior_c2), levels=0)
plt.scatter(c1_data[:, 0], c1_data[:, 1])
plt.scatter(c2_data[:, 0], c2_data[:, 1])
plt.title("Decision Boundary")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(["C1", "C2"])
plt.show()
```


![](https://i.imgur.com/Li4vvkU.png)


## Task 4: Non-parametric Density Estimation

#### a)

![](https://i.imgur.com/4AwE5zp.png)

The bin size for $\Delta 0.2$ looks a bit "spikey" and seems to overfit a bit (eg. the curve on the left goes a bit up again after it has reached its maximum).

For $\Delta 2.0$ the opposite seems to be the case as the result is very coarse.

The bin size for $\Delta 0.5$ performs best as each of its curves has clear maxima and minima. It's reminiscent of a mixture of Gaussians.

We cannot know whether this is the best bin size, however, because we do not know the real distribution.

```python
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
```

#### b)

![](https://i.imgur.com/tfjg5HF.png)

The above plot shows the kernel density estimates for the training data.

For these the log-likelihoods are:
```
L(0.03) =  -674.727938709064
L(0.2) =  -717.0216577444179
L(0.8) =  -795.6632833459042
```

For the log-likelihood, larger values (ie. values closer to 0) are better, so in this case $\sigma = 0.03$ yields the best result, which is kind counter-intuitive because from the plots one would think that it is quite noisy. $\sigma = 0.2$ seems to be a better choice.

The function V is wrongly named, but its functionality is as desired. The actual volume for the Gaussian kernel is 1.
```python
def kde(data):
    print("\t KDE")

    def V(h):
        d = 1
        return np.sqrt(2 * np.pi * h * h) ** d

    N = 1000
    x_vals = np.linspace(-4, 8, N)

    kde_1 = [gauss_kernel(x, 0.03, data) / (N * V(0.03)) for x in x_vals]
    kde_2 = [gauss_kernel(x, 0.2, data) / (N * V(0.2)) for x in x_vals]
    kde_3 = [gauss_kernel(x, 0.8, data) / (N * V(0.8)) for x in x_vals]

    plt.plot(x_vals, kde_1)
    plt.plot(x_vals, kde_2)
    plt.plot(x_vals, kde_3)
    plt.legend(["$\\sigma = 0.03$", "$\\sigma = 0.2$", "$\\sigma = 0.8$"])

    plt.title("Gaussian KDE")
    plt.xlabel("x")
    plt.ylabel("$p(x)$")
    plt.show()
```

The log-likelihoods are computed as follows:
```python
# Log-likelihoods
N_ll = len(data)
kde_1_ll = [gauss_kernel(x, 0.03, data) / (N_ll * gauss_norm_factor(0.03)) for x in data]
kde_2_ll = [gauss_kernel(x, 0.2, data) / (N_ll * gauss_norm_factor(0.2)) for x in data]
kde_3_ll = [gauss_kernel(x, 0.8, data) / (N_ll * gauss_norm_factor(0.8)) for x in data]
print("L(0.03) = ",np.sum(np.log(list(map(lambda x: 1e-11 if x == 0 else x, kde_1_ll)))))
print("L(0.2) = ", np.sum(np.log(list(map(lambda x: 1e-11 if x == 0 else x, kde_2_ll)))))
print("L(0.8) = ", np.sum(np.log(list(map(lambda x: 1e-11 if x == 0 else x, kde_3_ll)))))
```

#### c)

![](https://i.imgur.com/D4bgMFK.png)

The overall procedure is similar to the KDE estimator, except we use this function instead:
```python
def knn_kernel(x, K, data):
    N = len(data)
    V = np.sort(np.array([np.abs(x - x_i) for x_i in data]))
    return K / (N * 2 * V[K])
```

which is called like this:
```python
knn_1 = [knn_kernel(x, 2, data) for x in x_vals]
knn_2 = [knn_kernel(x, 8, data) for x in x_vals]
knn_3 = [knn_kernel(x, 35, data) for x in x_vals]
```

We can see that for smaller K the plot becomes really spikey. This is because the distances can become really small since we are only looking for nearby neighbors, so the small number in the nominator (`V[K]`) leads to a large result.
The result becomes smoother as we start accounting for more points, including those further away.


#### d)

The reason we test them on a different data set is to ensure that the estimators generalize well, ie. it is to avoid overfitting to the training examples.

All in all we get the following log-likelihoods:
```
== Train (KDE) ==========
L(0.03) = -674.727938709064
L(0.2)  = -717.0216577444179
L(0.8)  = -795.6632833459042

== Test (KDE) ===========
L(0.03) = -2812.1067563863708
L(0.2)  = -2877.32622829489
L(0.8)  = -3192.4847674164603

== Train (kNN) ==========
L(2)  = -572.035101237705
L(8)  = -691.1326697445517
L(35) = -711.1152008972026

== Test (kNN) ===========
L(2)  = -2343.934335395203
L(8)  = -2765.6253517746054
L(35) = -2830.8994901040096
```

Going by the log-likelihoods we should take $\sigma = 0.03$ for the KDE-estimator and $K=2$ for the kNN-estimator, as these are closest to 0.

Comparing these values to the plot, neither choice seems to be reasonable. 

For kNN the choice of $K=35$ could be seen as reasonable since for $K=8$ there still are a lot of spikes on the bigger curve. Also, we've tested higher values. For $K=200$ the left "bell" was completely smoothed out, and compared to that $K=35$ looks acceptable in our opinion as its shape is still reminiscent of two mixed Gaussians.
For the KDE-estimator a choice of $\sigma=0.2$ may be better assuming the underlying distribution is a mixture of Gaussians.


## Task 5: Expectation Maximization

#### a)

Our model's parameters are the means $\mu_k$, the covariances $\Sigma_k$, and the mixture weights $\pi_k$. Furthermore we have to keep track of the responsibilities $\alpha$.

The E-step: here we evaluate "\alpha" for every data point using the current parameters $\mu_k$, $\Sigma_k$, and $\pi_k$:

$$
\alpha_{nk} = \frac{\pi_k\mathcal{N}(\vec{x}_n | \vec{\mu}_k, \Sigma_k)}{\sum_j\pi_j\mathcal{N}(\vec{x}_n | \vec{\mu}_j, \Sigma_j)}
$$

The M-step: using the new $\alpha$ values, re-estimate the parameters (mean before covariance!):

\begin{align}
\mu_k &= \tfrac{1}{N_k} \sum_{n=1}^N \alpha_{nk} \vec{x}_n \\
\Sigma_k &= \tfrac{1}{N_k} \sum_{n=1}^N \alpha_{nk} (\vec{x}_n - \mu_k)(\vec{x}_n - \mu_k)^T \\
\pi_k &= \tfrac{N_k}{N}
\end{align}

Source:
[11.3 of Marc Deisenroth, Mathematics for Machine Learning]

#### b)

The main components of the algorithm (iteration, E- and M-step), look like this:

```python
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
```

```python
def Expectation(means, covs, mix_coefs, X):
    ''' Compute the posterior distribution for each mixture component and 
    for all data points'''
    all_we_need = [[mix_coef*multivariate_normal.pdf(x=x, mean=mean, cov=cov)
                    for mean, cov, mix_coef in zip(means, covs, mix_coefs)] for x in X]
    alphas = [[x/sum(xs) for x in xs] for xs in all_we_need]
    return alphas
```

```python
def Maximization(X, alpha_list):
    Nks = [sum(np.array(alpha_list)[:, i])
           for i in range(len(alpha_list[0]))]
    new_mix_coefs = [nk/len(X) for nk in Nks]
    new_means = np.array([[alpha*x for alpha in alphas]
                          for alphas, x in zip(alpha_list, X)])
    new_means = [sum(new_means[:, i])/Nks[i]
                 for i in range(len(new_means[0]))]
    # vec*vec.T
    new_cov = np.array([
    [alpha*np.matrix(x-new_mean).T*np.matrix((x-new_mean))
    for alpha, new_mean in zip(alphas, new_means)]
    for alphas, x in zip(alpha_list, X)])
    new_cov = [sum(new_cov[:, i])/Nks[i]
               for i in range(len(new_cov[0]))]
    return new_means, new_cov, new_mix_coefs
```

Below is a sample plot:

![](https://i.imgur.com/AzUUhh4.png)
![](https://i.imgur.com/tyJLEBN.png)
![](https://i.imgur.com/nuZfbgK.png)
![](https://i.imgur.com/JaJBzDQ.png)
![](https://i.imgur.com/cK0SiuZ.png)


The log-likelihoods for the mixture Gaussians are calculated as follows:

$$
\sum_{n=1}^N log (\sum_{k=1}^K \pi_k \mathcal{N}(\vec{x}_n | \vec{\mu}_k, \Sigma_k))
$$

where $\mathcal{N}$ represents a multivariate Gaussian:
```python
def multi_gauss(x, mean, cov):
    gauss = 0.0

    cov_det = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)

    norm = (1.0 / np.sqrt((2 * np.pi) ** 2 * cov_det))
    v = np.array(x - mean).reshape((2, 1))
    gauss += np.exp(-0.5 * v.T @ cov_inv @ v)
    gauss *= norm

    return gauss
```

The actual log-likelihood computation:
```python
log_likelihood = 0.0
for k in range(len(data)):
    ll = 0.0
    for idx in range(N):
        likelihood = mix_coefs[idx] * multi_gauss(data[k], means[idx], covs[idx])
        if likelihood == 0.0:
            likelihood = 1e-11
        ll += likelihood
    log_likelihood += np.log(ll).item()
log_likelihoods.append(log_likelihood)
```

which looks like this when plotted:

![](https://i.imgur.com/puouuwO.png)


Source:
[11.2 of Marc Deisenroth, Mathematics for Machine Learning]






