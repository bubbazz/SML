import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm,inv

noise = 0.005
interval = (0,2*np.pi)
steps = int((interval[1]-interval[0])/0.005)
x_steps = np.linspace(interval[0],interval[1],steps)
fig, axs = plt.subplots(5)
def plot_me(i,cov=False):
    if cov:
        plot_mu_cov(i)
    axs[i].plot(x_steps,function(x_steps))
    axs[i].scatter(x_lst,y_lst)
    #plt.show()
def plot_mu_cov(i):
    y1 = [(mu_f(x).reshape(-1) + 2*np.sqrt(cov_f(x).reshape(-1))).squeeze() for x in x_steps]
    #print(y1)
    y2 = [(mu_f(x).reshape(-1) - 2*np.sqrt(cov_f(x).reshape(-1))).squeeze()  for x in x_steps]
    axs[i].plot(x_steps,[mu_f(x).reshape(-1) for x in x_steps],label="mean")
    #axs[i].plot(x_steps,[cov_f(x).reshape(-1) for x in x_steps])
    axs[i].fill_between(x=x_steps, y1=y1, y2=y2, alpha=0.1,color="red",label="$\sigma$")
def kernel(xi,xj,sigma=0.5):
    return np.exp(-norm(xi-xj)**2/(2*sigma))
def function(x):
    return np.sin(x)+np.sin(x)**2
#INITAL
x_lst = [x_steps[steps//2]]
y_lst = [function(x_lst[0])]
#
mu = [0]
SIG1 = np.array([[kernel(x,y) for y in x_lst]for x in x_lst]).reshape(len(x_lst),len(x_lst)) + noise*np.eye(len(x_lst))
c = np.array([1+noise]).reshape(1,1)
#FIST
k = lambda y: np.array([kernel(y,x) for x in x_lst]).reshape(-1,1)
#mu_f = lambda x: np.array(mu) + k(x).T@inv(SIG1)@np.array(y_lst).reshape(-1,1)
# assumtion mu evertime 0; should be wrong because we need to update maybe just mu_new = 0
# or should be calced evertime (y - mu)
mu_f = lambda x: k(x).T@inv(SIG1)@np.array(y_lst).reshape(-1,1)
cov_f = lambda x: c - k(x).T@inv(SIG1)@k(x)
i = 1
j = 0
#color = ['r','g','b','y','c','m']
while i < 16:
    #print(f"mu {mu} \n x {x_lst} \n y {y_lst}")
    if i in [1,2,5,10,15]:
        plot_me(j,True)
        j += 1
    cov_lst = [np.sqrt(norm(cov_f(x).reshape(-1))) for x in x_steps]
    new_x = x_steps[np.argmax(cov_lst)]
    new_y = function(new_x)
    new_mu = mu_f(x_lst[-1])[0,0]
    x_lst.append(new_x)
    y_lst.append(new_y)
    mu.append(new_mu)
    SIG1 = np.array([[kernel(x,y) for y in x_lst]for x in x_lst]).reshape(len(x_lst),len(x_lst)) + noise*np.eye(len(x_lst))
    i += 1
#plt.label("mean","cov")
plt.show()
