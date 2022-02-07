from sklearn.svm import SVC
import os 
import numpy as np
import matplotlib.pyplot as plt
PATH = os.path.join('.','dataSets',"iris-pca.txt")
data = np.loadtxt(PATH)
X,y = data[:,0:2],data[:,2]

clf = SVC(kernel='linear')
clf.fit(X,y)
alphas = np.abs(clf.dual_coef_)
w = clf.coef_[0]
labels = np.sign(clf.dual_coef_)
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy
def print(X,y,w):
    X0,X1 = X[:,0],X[:,1]
    x = np.linspace(-2,2,10)
    xx, yy = make_meshgrid(X0, X1)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z)
    plt.scatter(X0,X1,c=y)
    plt.scatter([0],[0])
    plt.plot(x,-(w[0]/w[1])*x)
    plt.show()

print(X,y,w)