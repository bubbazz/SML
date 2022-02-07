import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean

data = np.loadtxt('./dataSets/ldaData.txt')
C1 = data[:50]
C2 = data[50:93]
C3 = data[93:]
C_LIST = [C1,C2,C3]

#LDA
def S_Class(X: np.array):
    meanX = np.mean(X,0)
    XX = X-meanX
    return sum([x.reshape(2,1)@x.reshape(1,2) for x in XX])
SW = sum([S_Class(C) for C in C_LIST])

SB = 0
overall_mean = np.mean(data,0)
for C in C_LIST:
    diffmean = (overall_mean-np.mean(C,0)).reshape(2,1)
    SB += len(C)*diffmean@diffmean.T
LDA_Eigen_L = [SW**0.5@SB@SW**0.5, np.linalg.inv(SW)@SB]
for LDA_Eigen in LDA_Eigen_L:
    eigenval,eigenvec = np.linalg.eig(LDA_Eigen)
    C1_DISTRIBUTION = C1@eigenvec
    C2_DISTRIBUTION = C2@eigenvec
    C3_DISTRIBUTION = C3@eigenvec
    C_DIST_LIST = [C1_DISTRIBUTION,C2_DISTRIBUTION,C3_DISTRIBUTION]
#for vec,val in zip(eigenvec,eigenval):
#   x = np.linspace(-2,2,1)
#   lst = np.array([i*vec for i in range(-2,10,1)])
#   plt.plot(lst[:,0],lst[:,1], label=f"$\lambda = {val}$")
#plt.legend()
i = 0
def foobar(C): # posterior(x)
    C = C[:,i]
    M =np.mean(C)
    COV = np.cov(C)
    prior = len(C)/len(data) 
    return (lambda x : np.exp(-((x-M)**2)/(2*COV))*prior/(np.sqrt(2*np.pi*COV)))
X = [foobar(C) for C in C_DIST_LIST]
for C in C_DIST_LIST:
    plt.scatter(C[:,0],C[:,1])
    weights = np.ones_like(C[:,0])/float(len(C[:,0]))
    plt.hist(C[:,0],weights=weights)
xs = np.linspace(-1,4,50)
for y,c in zip(X,["Blue","Orange","Green"]):
    plt.plot(xs,[y(x) for x in xs],color=c)
plt.title("FLDA Transformation w. Gauß distribution")
plt.xlabel("$v_{\lambda_1}$")
plt.ylabel("$v_{\lambda_2}$")
plt.show()
#
lst  = np.array([[y((c@eigenvec)[i]) for y in X] for c in data])
arg_lst = np.argsort(lst)[:,-1]
C1_pred = np.array([d for d,i in zip(data,arg_lst) if i == 0])
C2_pred = np.array([d for d,i in zip(data,arg_lst) if i == 1 ])
C3_pred = np.array([d for d,i in zip(data,arg_lst) if i == 2 ])
#
for i,C,CP,c in zip(range(1,4,1),[C1,C2,C3],[C1_pred,C2_pred,C3_pred],["Blue","Orange","Green"]):
    plt.scatter(C[:,0],C[:,1],marker=".",color=c)
    plt.scatter(CP[:,0],CP[:,1],marker="x",color=c)
plt.legend(["real","predicted"])
plt.title("real vs. predicted classes")
plt.show()

