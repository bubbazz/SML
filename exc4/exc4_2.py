import os
import cvxopt
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix as cvxopt_matrix
##
from sklearn.svm import SVC

PATH = os.path.join('.','exc4','dataSets',"iris-pca.txt")
data = np.loadtxt(PATH)

X,y = data[:,0:2],data[:,2]
def y_classifier(y):
    return np.array([1 if yi == 2 else -1 for yi in y ])
###
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy
###
def print_x(X,y,w,b,S):
    x = np.linspace(-2,2,10)
    #clf = SVC(C=10,kernel='linear',)
    #clf.fit(X,y)
    #xx,yy = make_meshgrid(X[:,0],X[:,1])
    #Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    #Z = Z.reshape(xx.shape)
    #plt.contourf(xx, yy, Z)
    plt.scatter(X[:,0],X[:,1],c=y,)
    plt.scatter(X[S][:,0],X[S][:,1],marker="x")
    plt.plot(x,(-w[0]/w[1])*x+b,)
    plt.show()
def SVM_normal(X:np.array,y:np.array,C=10):
    y = y.reshape(100,1) * 1.
    m,n = X.shape
    Xy = y.reshape(-1,1 ) * X
    H = Xy@Xy.T
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))
    #G = cvxopt_matrix(-np.eye(m))
    G = cvxopt_matrix(np.vstack((np.eye(m)*-1.,np.eye(m))))
    #h = cvxopt_matrix(np.zeros(m))
    h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))
    solver = cvxopt.solvers.qp(P,q,G,h,A,b,)
    alphas = np.array(solver['x'])
    w = ((y * alphas).T @ X).reshape(-1,1)
    S = (alphas > 1e-4).flatten()
    SS = (alphas > 1e-3).flatten()
    b = sum(y[S] - np.dot(X[S], w))/len(y[S])
    print(f"alphas not null {len(alphas[S])}")
    print('Alphas = ',alphas)
    print('w = ', w.flatten())
    print(f'b = {b}')
    return w,b,SS
def SVM_kernel(X:np.array,y:np.array):
    raise NotImplementedError
y = y_classifier(y)
w,b,S = SVM_normal(X,y)
print_x(X,y,w,b,S)
