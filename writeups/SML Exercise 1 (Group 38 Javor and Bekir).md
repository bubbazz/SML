# SML Exercise 1 (Group 38: Javor and Bekir)
## Task 1: Model Fitting

#### Model 1
![](https://i.imgur.com/sHx0AgM.png)

Under this model the triangle will be labeled as a striped circle:
- Testing accuracy: 1/1 correct
- Training accuracy: 9/11 filled circles correct, 9/10 striped circles correct

#### Model 2
![](https://i.imgur.com/17LsDDi.png)

Under this model the triangle will be labeled as a filled circle:
- Testing accuracy: 0/1 correct
- Training accuracy: 9/11 filled circles correct, 7/10 striped circles correct

$\rightarrow$ Overall Model 1 has a better training and testing accuracy, so it should be preferred over Model 2.


## Task 2: Linear Algebra Refresher

### 2.a Matrix Properties
#### Addition
The defintion of matrix addition:
![](https://i.imgur.com/NusDbiA.png)

Proof of commutativity:
![](https://i.imgur.com/YFWvr4i.png)

Proof of associativity:
![](https://i.imgur.com/i3V1LLU.png)

#### Multiplication
Matrix multiplication is defined as follows:
$$ 
A\cdot B = C \Leftrightarrow c_{ij} = \sum_k a_{ik} \cdot b_{kj}
$$

Or in element-wise notation, where $A_{ij}$ denotes the entry in row $i$ and column $j$ of the (here:) $n \times n$ matrix $A$:
$$
    (AB)_{ij} = a_{i1}b_{1j} + \dots + a_{in}b_{nj}.
$$

Proof that the multiplication is not commutative:
$$
\left(\begin{smallmatrix} 0 & 1 \\ 0 & 0 \end{smallmatrix}\right) 
\cdot 
\left(\begin{smallmatrix} 1 & 0 \\ 0 & 0 \end{smallmatrix}\right) = 
\left(\begin{smallmatrix} 0 & 0 \\ 0 & 0 \end{smallmatrix}\right) 
\neq 
\left(\begin{smallmatrix} 0 & 1 \\ 0 & 0 \end{smallmatrix}\right) =
\left(\begin{smallmatrix} 1 & 0 \\ 0 & 0 \end{smallmatrix}\right)
\cdot
\left(\begin{smallmatrix} 0 & 1 \\ 0 & 0 \end{smallmatrix}\right) 
$$


Proof of associativity:

\begin{align}
  (A(BC))_{ij} &= a_{i1}(b_{11}c_{1j}+\dots b_{1n}c_{nj}) + \dots + a_{in}(b_{n1}c_{1j}+\dots b_{nn}c_{nj}) \\
      &= (a_{i1}b_{11}c_{1j} + \dots + a_{i1}b_{1n}c_{nj}) + \dots + (a_{in}b_{n1}c_{1j} + \dots + a_{in}b_{nn}c_{nj}) \\
      &= (a_{i1}b_{11})c_{1j} + \dots + (a_{i1}b_{1n})c_{nj} + \dots + (a_{in}b_{n1})c_{1j} + \dots + (a_{in}b_{nn})c_{nj} \\
      &= (a_{i1}b_{11} + \dots + a_{in}b_{n1})c_{1j} + \dots + (a_{i1}b_{1n} + \dots + a_{in}b_{nn})c_{nj} \\
      &= ((AB)C)_{ij}
\end{align}

#### Both

Proof of distributivity (left):
\begin{align}
    (A(B + C))_{ij} &= a_{i1}(b_{1j}+c_{1j}) + \dots 
                        + a_{in}(b_{nj}+c_{nj}) \\
                    &= (a_{i1}b_{1j}+a_{i1}c_{1j}) + \dots 
                        + (a_{in}b_{nj}+a_{in}c_{nj}) \\
                    &= (a_{i1}b_{1j}+\dots+a_{in}b_{nj})
                        + \dots
                        + (a_{i1}c_{1j}+\dots+a_{in}c_{nj}) \\
                    &= (AB + AC)_{ij}
\end{align}

Proof of distributivity (right):
\begin{align}
    ((A + B)C)_{ij} &= (a_{i1}+b_{i1})c_{1j} + \dots 
                        + (a_{in}+b_{in})c_{nj} \\
                    &= (a_{i1}c_{1j}+b_{i1}c_{1j}) + \dots 
                        + (a_{in}c_{nj}+b_{in}c_{nj}) \\
                    &= (a_{i1}c_{1j}+\dots+a_{in}c_{nj})
                        + \dots
                        + (b_{i1}c_{1j}+\dots+b_{in}c_{nj}) \\
                    &= (AC + BC)_{ij}
\end{align}


### 2.b Matrix Inversion
A Inverse can be calculated with the Gaußian elemination
$$
\left[\begin{array}{@{}ccc|ccc@{}}
    1 & a & b & 1 & 0 & 0 \\
 1 & c & d & 0 & 1 & 0\\
 0 & 0 & 1 & 0 & 0 & 1
\end{array}\right]
$$
II - I
$$
\left[\begin{array}{@{}ccc|ccc@{}}
     1 & a & b & 1 & 0 & 0 \\
     0 & c-a & d-b & -1 & 1 & 0\\
     0 & 0 & 1 & 0 & 0 & 1
\end{array}\right]
$$
II -(d-b)III & I-bIII
$$
\left[\begin{array}{@{}ccc|ccc@{}}
     1 & a & 0 & 1 & 0 & -b \\
     0 & c-a & 0 & -1 & 1 & b-d\\
     0 & 0 & 1 & 0 & 0 & 1
\end{array}\right]
$$
II/(c-a)
$$
\left[\begin{array}{@{}ccc|ccc@{}}
     1 & a & 0 & 1 & 0 & -b \\
     0 & 1 & 0 & \frac{-1}{c-a} & \frac{1}{c-a} & \frac{b-d}{c-a}\\
     0 & 0 & 1 & 0 & 0 & 1
\end{array}\right]
$$
I - aII 
$$
\left[\begin{array}{@{}ccc|ccc@{}}
     1 & 0 & 0 & 1+\frac{a}{c-a} & -\frac{a}{c-a} & -b-\frac{a(b-d)}{c-a} \\
     0 & 1 & 0 & \frac{-1}{c-a} & \frac{1}{c-a} & \frac{b-d}{c-a}\\
     0 & 0 & 1 & 0 & 0 & 1
\end{array}\right]
$$
$c - a \neq 0$ is the only condition that has to hold so that $A_1$ has an inverse matrix.

(Other $A$): To check if $A$ is invertible, we can compute its determinant and check that $det(A) \neq 0$:

\begin{equation*}
    det(A) = 1 \cdot det(\begin{smallmatrix}
        2 & 3 \\
        8 & 12
    \end{smallmatrix}) = 1 \cdot (24 - 24) = 0
\end{equation*}

Hence the inverse does not exist.


### 2.c Matrix Pseudoinverse
- Pay attention to the constraints: full row/column rank, see <https://www.cds.caltech.edu/~murray/amwiki/index.php/FAQ:_What_does_it_mean_for_a_non-square_matrix_to_be_full_rank%3F#:~:text=For%20a%20non%2Dsquare%20matrix%20with%20rows%20and%20columns%2C%20it,the%20shape%20of%20the%20matrix.>

- Left:  $A^{\#}A = (A^TA)^{-1}A^TA$ (works if $A$ has full column rank)
- Right: $AA^{\#} = AA^T(AA^T)^{-1}$ (works if $A$ has full row rank)


Given that $A$ is a $2 \times 3$ matrix (ie. more columns than rows), its rows are linearly independent, meaning it has full row rank. This implies that the right pseudo-inverse can be computed.

- $A \in \Re^{2 \times 3}$
- $A^T \in \Re^{3 \times 2}$
- $AA^T \in \Re^{2 \times 2}$
- $(AA^T)^{-1} \in \Re^{2 \times 2}$
- $AA^T(AA^T)^{-1} = I_2 \in \Re^{2 \times 2}$


### 2.d Basis Transformation
The transformation matrix $T$ is spanned by $b_1$ and $b_2$:
$$
T=\begin{pmatrix}
2 & 3 \\
3 & 4 
\end{pmatrix}
$$

This gives us $Tv 
= T \left( 
\begin{smallmatrix} 
    2 \\ 5 
\end{smallmatrix} \right) 
= \left( 
\begin{smallmatrix} 
    19 \\ 26 
\end{smallmatrix} \right)$. 

## Task 3: Statistics Refresher
### 3.a.1 Generic Questions
#### Definitions
The expection for a function f is defined as:
\begin{equation*}
E[f] =
\begin{cases}
    \sum_x & f(x)p(x) & \text{if discrete} \\
    \int_x & f(x)p(x)dx & \text{if continuous}
\end{cases}
\end{equation*}

The variance is defined as:
$$var[f] = E[(f - E[f])^2] = E[f^2]-E[f]^2$$

#### Linear Operators
A linear operator is a mapping from one space to another such that the following two condtions hold ($\lambda$ is a scalar):
$$
\begin{align}
M(x+y) &= M(x) + M(y)\\
M(\lambda x) &= \lambda M(x).
\end{align}
$$

$E$ is linear because $E[\alpha x] = \alpha E[x]$ and $E[x+y] = E[x] + E[y]$ (see lecture 3, slide 39).

Proof for the expection value (if needed):

$E[f+h] = \sum_x(f(x)+h(x))p(x) = \sum_xf(x)p(x)+h(x)p(x)=\sum_xf(x)p(x)+\sum_xh(x)p(x) = E[f]+E[h]$

$E[\lambda f] = \sum_x\lambda f(x)p(x) = \lambda \sum_x f(x)p(x) = \lambda E[f]$

For the variance we have (using the fact that $E$ is linear)

\begin{align}
var[\alpha x] &= E[(\alpha x)^2] - E[\alpha x]^2 \\
              &= E[\alpha^2 x^2] - \alpha^2 E[x]^2 \\
              &= \alpha^2 E[x^2] - \alpha^2 E[x]^2 \\
              &= \alpha^2 (E[x^2] - E[x]^2) \\
              &= \alpha^2 var[x] \\
              & \neq \alpha \cdot var[x]
\end{align}


Hence the variance is not a linear operation.


### 3.a.2  Dices
#### General
The domain of random variable $\Omega := \{ 1,2,3,4,5,6 \}$ and $X_i = id$
#### Expections
The mean function $\bar{X} = \frac{\sum^N_i x_i}{N}$ is an unbiased estimator it can be rewitten as following, (TODO sollten wir ein beweisen hier rein schreiben oder einfach annehmen)
- where $N$ is the number of dicerolls
- and $p(x_i) = \frac{1}{n}$ for readablity we factorised it 
$$
\begin{align}
E[X] &= \sum_{x \in A} x_i p(x_i) \\&=
1 (\frac{1}{18}) + 2 (\frac{5}{18}) + 3 (\frac{6}{18}) + 4(\frac{3}{18}) + 5(\frac{2}{18}) + 6(\frac{1}{18})
\\&= \frac{1+10+18+12+10+6}{18} = \frac{57}{18} = 3,17
\\\\
E[X] &= \sum_{x \in B} x_i p(x_i) \\
&=1 (\frac{6}{18}) + 2 (\frac{1}{18}) + 3 (\frac{1}{18}) + 4(\frac{4}{18}) + 5(\frac{1}{18}) + 6(\frac{5}{18}) \\
&= \frac{6+2+3+16+5+30}{18} = 3,44
\\\\
E[X] &= \sum_{x \in C} x_i p(x_i) \\
&= 1 (\frac{3}{18}) + 2 (\frac{2}{18}) + 3 \frac{3}{18}) + 4(\frac{3}{18}) + 5(\frac{4}{18}) + 6(\frac{3}{18})
\\
&=\frac{3+4+9+12+20+18}{18} = 3.67  
\end{align}
$$

#### Variance 
To calculate the variance, we also need $E[X^2]$:


\begin{align}
E[A^2] &= \frac{1}{18}(1 \cdot 1^2 + 5 \cdot 2^2 + 6 \cdot           3^2 + 3 \cdot 4^2 + 2 \cdot 5^2 + 1 \cdot 6^2) \\
       &= \frac{209}{18} \\
       &= 11.61
\end{align}

\begin{align}
E[B^2] &= \frac{1}{18}(6 \cdot 1^2 + 1 \cdot 2^2 + 1 \cdot           3^2 + 4 \cdot 4^2 + 1 \cdot 5^2 + 5 \cdot 6^2) \\
       &= \frac{288}{18} \\
       &= 16
\end{align}

\begin{align}
E[C^2] &= \frac{1}{18}(3 \cdot 1^2 + 2 \cdot 2^2 + 3 \cdot           3^2 + 3 \cdot 4^2 + 4 \cdot 5^2 + 3 \cdot 6^2) \\
       &= \frac{294}{18} \\
       &= 16.33
\end{align}

Now we can calculate the variances:

\begin{align}
var[A] &= E[A^2] - E[A]^2 = 11.61 - 3.17^2 &= 1.56 \\
var[B] &= E[B^2] - E[B]^2 = 16    - 3.44^2 &= 4.17 \\
var[C] &= E[C^2] - E[C]^2 = 16.33 - 3.67^2 &= 2.86 \\
\end{align}

#### KL-Divergence

For the uniform distribution we have $q(x_i) = 3, i=1,\dots,6$.

\begin{align}
    KL(A||q) &= -(1ln3 + 5ln\frac{3}{5} + 6ln\frac{3}{6} +                     3ln1 + 2ln\frac{3}{2} + 1ln3) &= 3.7 \\
    KL(B||q) &= -(6ln\frac{3}{6} + 1ln3 + 1ln3 + 4ln\frac{3} {4} + 1ln3 + 5ln\frac{3}{5}) &= 4.6 \\
    KL(C||q) &= -(3ln1 + 2ln\frac{3}{2} + 3ln1 + 3ln1 +                         4ln\frac{3}{4} + 3ln1) &= 0.3 \\
\end{align}

$C$ is the closest to being a fair die.


### 3b It is a Cold World 


\begin{align}&Sickness_1(\omega) = 
\begin{cases}
1 &\quad\text{if }\omega\text{ = Cold } \\
-1 &\quad\text{if }\omega\text{ = }\lnot \text{Cold } \\
\end{cases} \\
&Sickness_2(\omega) = 
\begin{cases}
1 &\quad\text{if }\omega \text{ = Pain} \\
-1 &\quad\text{if }\omega\text{ = }\lnot \text{Pain } 
\end{cases} \\
&\Omega_1 = \{Cold,\lnot Cold\} \\
&\Omega_2 = \{Pain,\lnot Pain\} \\
\end{align}

*  A person with a cold has back pain 25% of the time.

$$
P(Sickness_2 = Pain | Sickness_1 = Cold) = 0.25
$$

* 4% of the world population has a cold.

$$
P(Sickness_1 = Cold) = 0.04
$$

*  10% of those who do not have a cold still have back pain
$$
P(Sickness_2 = Pain | Sickness_1 = \lnot Cold) = 0.1
$$

* If you suffer from back pain, what are the chances that you suffer from a cold?

With the rule of Bayes $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$ and $P(B) = \sum_{i \in [0,|\Omega|]} P(B|A_i)P(A_i)$ we can reverse calculate this informations 

$$
\begin{align}
&P(Sickness_2 = Pain) \\&= P(Sickness_2 = Pain | Sickness_1=Cold)P(Sickness_1=Cold)+P(Sickness_2 = Pain | Sickness_1 = \lnot Cold)P(Sickness_1=\lnot Cold)\\&= 0.1\cdot 0.96 + 0.25 \cdot 0.04 =0.106 \\\\
&P(Sickness_1 = Cold | Sickness_2 = Pain) \\
&= \frac{P(Sickness_2 = Pain | Sickness_1 = Cold)P(Sickness_1 = Cold)}{P(Sickness_2 = Pain)} = \frac{0.25\cdot 0.04}{0.106} = 0.0943
\end{align}
$$

So the probability for having a cold given backpain is 9.43%

### 3.c  Cure the Virus


#### 1
Markov Model as Graph:
![](https://i.imgur.com/LUT9nd0.png)


We can model this as a system of linear equations:
\begin{align}
m_{t+1} &= 0.42m_t + 0.026\tilde{m}_t \\
\tilde{m}_{t+1} &= 0.58m_t + 0.974\tilde{m}_t
\end{align}

Writing this in matrix notation
$$
M =
\begin{pmatrix}
      0.42 & 0.026 \\
      0.58 & 0.974
\end{pmatrix}
$$

allows us to express the iteration step as follows:
$$
    s_{t+1} = M \cdot s_{t}
$$

#### 2

The probability that the mutation will still be present after 18 days is about 4.29%. The code and the plot is listed below.

```python
import numpy as np
from matplotlib import pyplot as plt

markov_chain_matrix = np.array([[0.42, 0.026], [0.58, 0.974]])
s0 = [1, 0]
slist = [s0]
n = 18
for i in range(n):
    print(i, s0)
    s0 = markov_chain_matrix.dot(s0)
    slist.append(s0)
print(n+1, s0)
plt.plot(range(n+1), [x[0] for x in slist])
plt.plot(range(n+1), [x[1] for x in slist])
plt.legend(['$m$', '$\\tilde{m}$'])
plt.show()

```
![](https://i.imgur.com/fA528FI.png)

#### 3

We ran the simulation for 100 days. After 11 days their is no more significant change, and after 21 days there is practically no change in the probabilities.

Below is the output for $M^{d} \cdot \vec{s}_0$ with $\vec{s}_0 = (1,0)^T$ as well as the corresponding plot.
```
0 [1, 0]
1 [0.42 0.58]
2 [0.19148 0.80852]
3 [0.10144312 0.89855688]
4 [0.06596859 0.93403141]
5 [0.05199162 0.94800838]
6 [0.0464847 0.9535153]
7 [0.04431497 0.95568503]
8 [0.0434601 0.9565399]
9 [0.04312328 0.95687672]
10 [0.04299057 0.95700943]
11 [0.04293829 0.95706171]
12 [0.04291768 0.95708232]
13 [0.04290957 0.95709043]
14 [0.04290637 0.95709363]
15 [0.04290511 0.95709489]
16 [0.04290461 0.95709539]
17 [0.04290442 0.95709558]
18 [0.04290434 0.95709566]
19 [0.04290431 0.95709569]
20 [0.0429043 0.9570957]
21 [0.04290429 0.95709571]
...
99 [0.04290429 0.95709571]
100 [0.04290429 0.95709571]
```
![](https://i.imgur.com/hb3485x.png)


To see whether the Markov chain actually converges to this solution, we can check whether this is its stationary distribution.

From [1, Theorem 17.2.2]: 
*Every irreducible (singly connected), ergodic Markov chain has a limiting distribution, which is equal to π, its unique stationary distribution.*

Looking at our graph, we can see that these properties hold:
- **Irreducible**: we can get from every state into every other state
- **Recurrent**: a state is recurrent if the Markov chain will return to it
- **Non-null recurrent**: the expected time to return to a state is finite 
- **Aperiodic**: the number of steps for a chain starting at and returning to state *i* must be divisible only by 1 (given in our case becaue all states have a loop, see [1])
- **Ergodic**: aperiodic + recurrent + non-null

In our case we have $\pi = (0.04290429, 0.95709571)^T$. So the chances that the mutation will still be present in the long run is indeed 4.29%.

To confirm this, we can also check whether the following holds for an aperiodic and irreducible Markov chain:

$$
\lim_{n \to \infty} P^{n} = (1 ,..., 1)^T \vec{\pi} \\
$$

Ie. the rows of the matrix converge to the (normalized) eigenvector $\vec{\pi}$ with eigenvalue 1.

In our case we have

\begin{align}
& M \vec{v} = \lambda \vec{v} &&(1) \\
& \vec{\pi} = \tfrac{\vec{v}}{||v||_1} \\
&\lim_{n \to \infty} M^{n} = (1, 1)^T \vec{\pi} && (2)
\end{align}

which is also printed below.

```python=
def eigenvalues():
    markov_chain_matrix = np.array([[0.42, 0.026], [0.58, 0.974]])
    print(markov_chain_matrix, markov_chain_matrix.T)
    lamb, vec = np.linalg.eig(markov_chain_matrix)
    print("eigenvalue_1 :" + str(lamb[1]) +
          "\neigenvector_1 : " + str(vec[:, 1]))
    # analitically calculated eigenvectors
    wolfram_mat = [13/290, 1]
    stochastical_norm = sum([x for x in wolfram_mat])
    wolfram_mat_normed = (1/stochastical_norm)*np.array(wolfram_mat)
    print("normed calc:" + str(wolfram_mat_normed))
    markov_chain_matrix = markov_chain_matrix.T
    n = 100
    x = np.linalg.matrix_power(markov_chain_matrix, n)
    print(n, x)
```
```shell=
normed calc:[0.04290429 0.95709571]
10000 [[0.04290429 0.95709571]
 [0.04290429 0.95709571]]
```

See: 
- [1]: [Kevin Murphy - Machine Learning, A Probabilistic Perspective]
- https://de.wikipedia.org/wiki/Station%C3%A4re_Verteilung
- https://brilliant.org/wiki/markov-chains/
- https://brilliant.org/wiki/stationary-distributions/

## Task 4: Information Theory

(Shannon's) entropy is defined as $-\sum_i p_i log_2(p_i)$ where $p_i \in [0, 1]$.


#### 1

The entropy is $-\sum_i S_i log_2(S_i) = 1.32$ .

#### 2

Using a set of four symbols, the maximum number of bits that can be transmitted is $log_2(4) = 2$. A uniform distribution is needed for this, ie. $S_i = 0.25$. In this case we would get $-(4 \cdot \tfrac{1}{4} log_2(\tfrac{1}{4})) = 2$.



## Task 5: Bayesian Decision Theory

#### a) Optimal Boundary
Bayesian decision theory is a statistical approach to the problem of pattern classification. The goal is to make predictions about unforeseen events in the future, given some known obersavations. 

If $P(C_1|x) > P(C_2 | x)$ choose $C_1$, else $C_2$ if $P(C_2|x) > P(C_2 | x)$. For equality you can have several options:
* random 
* pick the class with more samples
* ...

At an optimal decision boundary equality holds.

#### b) Decision Boundaries
Given:
$$
P(C_1) = P(C_2) \\
P(x|C_i) = P(x|\mu_i, \sigma_i)\\
\sigma = \sigma_i , i\in [1,2]
$$

Furthermore, we know that
$$
P(x|\mu_i, \sigma_i) = \tfrac{1}{\sqrt{2\pi\sigma_i^2}}exp(-\tfrac{(x-\mu_i)^2}{2\sigma^2})
$$

and that the optimal decision boundary is given when $P(C_1| x ) = P(C_2 | x)$. So our goal is to find an $x^{*}$ for which this condition holds.

First we make use of some equivalences:
\begin{align}
P(C_1| x ) &= P(C_2 | x) &&| \text{Bayes theorem} \\
\frac{P(x|C_1)P(C_1)}{P(x)} &= \frac{P(x|C_2)P(C_2)}{P(x)}  &&| \cdot P(x), P(C_1) = P(C_2)\\
P(x | C_2) &= P(x | C_1) \\
\end{align}

Using $P(x|C_i) = P(x|\mu_i, \sigma_i)$ and $\sigma = \sigma_1 = \sigma_2$ we can solve the Gaussian distribution for $x^{*}$ at the optimal decision boundary:

\begin{align}
 P(x|\mu_1, \sigma) &= P(x|\mu_2, \sigma) \\
 \tfrac{1}{\sqrt{2\pi\sigma^2}}exp({-\tfrac{(x-\mu_1)^2}{2\sigma^2}}) &= \tfrac{1}{\sqrt{2\pi\sigma^2}}exp(-{\tfrac{(x-\mu_2)^2}{2\sigma^2}}) &&|\cdot \sqrt{2\pi\sigma^2} \\
 exp({-\tfrac{(x-\mu_1)^2}{2\sigma^2}}) &= exp(-{\tfrac{(x-\mu_2)^2}{2\sigma^2}}) &&| \cdot ln \\
 -\tfrac{(x-\mu_1)^2}{2\sigma^2} &= -\tfrac{(x-\mu_2)^2}{2\sigma^2} &&| \cdot 2\sigma^2 \\
 -(x-\mu_1)^2 &= -(x-\mu_2)^2 &&| \dots \\
 x &= \tfrac{\mu_1 + \mu_2}{2}
\end{align}

So we have $x^{*} = \tfrac{\mu_1 + \mu_2}{2}$.


#### c) Different Misclassification Costs

*  $P(C_1) = P(C_2)$ and $\sum P(C_i) = 1$ follows that $P(C_1) = P(C_2) = \frac{1}{2}$ . 
* $\sigma_1 = \sigma_2 = \sigma$ 

Let $\alpha_1$ = labeling $x$ as $C_1$ and let $\alpha_2$ = labeling $x$ as $C_2$.
Given loss function $\lambda(\text{choice}, \text{true value})$, we have:

\begin{align}
\lambda_{11} &= \lambda(\alpha_1, C_1) \\
\lambda_{12} &= \lambda(\alpha_1, C_2) \\
\lambda_{21} &= \lambda(\alpha_2, C_1) \\
\lambda_{22} &= \lambda(\alpha_2, C_2) \\
\end{align}

Using  $\lambda_{12} = 4\lambda_{21}$ and slide 29:

$$
\tfrac{\lambda_{21} - \lambda_{11}}{4\lambda_{21} - \lambda_{22}} = \tfrac{P(x|C_2)P(C_2)}{P(x|C_1)P(C_1)}
$$

Since $P(C_1) = P(C_2)$ we can further simplify the right fraction:

$$
\tfrac{\lambda_{21} - \lambda_{11}}{4\lambda_{21} - \lambda_{22}} = \tfrac{P(x|C_2)}{P(x|C_1)},
$$

and because there is no cost for correct classifications, ie. $\lambda_{11} = \lambda_{22} = 0$, we can also simplify the left side:

$$
\tfrac{\lambda_{21}}{4\lambda_{21}} = \tfrac{P(x|C_2)P(C_2)}{P(x|C_1)P(C_1)},
$$


and simplifying the left even further:

$$
\tfrac{1}{4}= \tfrac{P(x|C_2)}{P(x|C_1)}.
$$

With the condtions $\mu_1 > 0$ , $\mu_1 = 2\mu_2$ we can solve for $x$:
$$
\tfrac{1}{4} = \tfrac{P(x|\tfrac{\mu}{2}, \sigma)}{P(x|\mu,\sigma)} \\
\tfrac{1}{4}=\tfrac{\tfrac{1}{\sqrt{2\pi\sigma^2}}exp(-\tfrac{(x-\tfrac{\mu}{2})^2}{2\sigma^2})}{\tfrac{1}{\sqrt{2\pi\sigma^2}}exp(-\tfrac{(x-\mu)^2}{2\sigma^2})} \\
\tfrac{1}{4}=exp(-\tfrac{(x-\tfrac{\mu}{2})^2}{2\sigma^2})\cdot exp(\tfrac{(x-\mu)^2}{2\sigma^2}) \\
\tfrac{1}{4}=exp(-\tfrac{(x-\tfrac{\mu}{2})^2}{2\sigma^2}+\tfrac{(x-\mu)^2}{2\sigma^2})\\
ln(\tfrac{1}{4})2\sigma^2 = (x-\mu)^2-(x-\tfrac{\mu}{2})^2 \\
x = \tfrac{3 \mu}{4} + 4 \tfrac{\sigma^2}{\mu} ln(2)
$$