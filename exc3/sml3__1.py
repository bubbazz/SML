import numpy as np 
import math
import matplotlib.pyplot as plt

# Implement linear ridge regression using linear features

data_test = np.loadtxt('./dataSets/lin_reg_test.txt')
x_test = data_test[:,0]
y_test = data_test[:,1]
data_train = np.loadtxt('./dataSets/lin_reg_train.txt')
x_train = np.array(data_train[:,0])
y_train = np.array(data_train[:,1]).reshape(len(data_train[:,1]),1)


# Derive the optimal model parameters by minimizing the squared error loss function
def linRegMSE(lamb , X : np.matrix, y : np.matrix):

    return np.linalg.inv(X @ X.T - lamb*np.identity(len(X))) @ (X @ y)
def plot(X,y,w):
    plt.scatter(X,y)
def RootMeanSqaredError(X:np.array,y:np.array, w : np.array):
    return math.sqrt(sum((y - X.T @ w)**2)/len(y))
x_train = np.column_stack((x_train,np.ones_like(x_train))).T# 50,2 2,50
w = linRegMSE(0.01,x_train,y_train)
print(w.shape,x_train.shape)
error = RootMeanSqaredError(x_train,y_train,w)
print(error)


