import numpy as np
import os
def pred(X, Y):
    N = len(X)
    pred = np.empty(N)  # Correct predictions
    for i in range(N):
        x = X[i]
        y = Y[i]
        out,_ = forwardPropgation(x)
        pred[i] = 1 if np.argmax(out[-1]) == np.argmax(y) else 0
    acc = np.sum(pred) / float(N)
    print("ACC = ", acc)
    print("PRED: ", pred.shape)

def shuffle_data( X, Y):
    indices = np.random.permutation(np.arange(X.shape[0]))
    X = X[indices, :]
    Y = Y[indices, :]
    return X, Y
PATH = os.path.join('.','exc4','dataSets')
network = { 'weigth' : [],
            'dweigth' : [],
            'activition' : [] ,
            'activitionDer' : [],
            'loss': [] ,
            'lossDer': [],
            'lr': 0.5}

def loadData(path:str):
    X_test = np.loadtxt(os.path.join(path,'mnist_small_test_in.txt'),delimiter=',', dtype =np.float64)
    y_test = np.loadtxt(os.path.join(path,'mnist_small_test_out.txt'),dtype =np.float64)
    X_train = np.loadtxt(os.path.join(path,'mnist_small_train_in.txt'),delimiter=',', dtype =np.float64)
    y_train = np.loadtxt(os.path.join(path,'mnist_small_train_out.txt'),dtype =np.float64)
    return X_train,y_train,X_test,y_test
## Layer Gen
def OneHotEncoding(y):
    n = len(np.unique(y))
    out = []
    for yi in y:
        zeros = np.zeros((n,1))
        zeros[int(yi),0] = 1
        out.append(zeros)
    return np.array(out,dtype =np.float64)
def BuildNN(hidden_layer_lst,hidden_act_lst):
    network['activition'] = [ActivationFunc(name) for name in hidden_act_lst]
    network['activitionDer'] = [ActivationFuncDer(name) for name in hidden_act_lst]
    network['weigth'] = HiddenWeightGen(hidden_layer_lst)
    network['dweigth'] = network['weigth'].copy()
# Better Bulding
def HiddenWeightGen(dimlist, low = -1,high=1):
    return [np.random.uniform(low,high,(dimlist[i],dimlist[i-1])).astype(np.float64) for i in range(1,len(dimlist))]
#Output-/Input-Layer build in fit
#    T = np.unique(y)
# Activierunsfuntionen 
def ActivationFunc(name:str = 'linear'):
    func = None
    if name == 'linear' : 
        func = (lambda x : x)
    elif name == 'sigmoid': 
        func = lambda x : (1/(1+np.exp(-x)))
    elif name == 'ReLU' : 
        func = lambda x : np.array([max(0,xi) for xi in x])
    elif name == 'tanh' : 
        func = lambda x : np.tanh(x)
    elif name == 'softmax': 
        #stable version
        func = lambda x : (np.exp(x-np.max(x))/np.sum(np.exp(x-np.max(x))))
        #func = lambda x : (np.exp(x)/np.sum(np.exp(x)))
    return func
def ActivationFuncDer(name:str = 'linear'):
    derv = None
    if name == 'linear' :   derv = lambda x :np.ones_like(x)
    elif name == 'sigmoid' :   derv = sigmoid_back
    elif name == 'ReLU':      derv = lambda x : np.array([1 if xi > 0 else 0 for xi in x])
    elif name == 'softmax':  derv = lambda x : np.ones_like(x) #softmax_back
    return derv
# TODO check
def sigmoid_back(x):
    sigmoid = ActivationFunc('sigmoid')(x)
    return sigmoid*(1-sigmoid)
def softmax_back(x):
    #print('soft')
    #x = np.clip(x, -709.78, 709.78) ## ,makes convergences slow 
    softmax = ActivationFunc('softmax')(x)
    return softmax * np.identity(softmax.size) - softmax.T @ softmax   
def ErrorFunction(name:str = 'MSE'):
    if name == 'MSE' : return lambda y,t : 0.5*np.sum((y-t)**2)
    elif name == 'altMSE': return lambda y,t : (y-t)**2
    # hmm https://stats.stackexchange.com/questions/198038/cross-entropy-or-log-likelihood-in-output-layer says LL
    # should be logliky but didnt care 
    elif name == "Xentropy" : return lambda y,t: -np.log(y[(t==1).flatten()]+1e-10) 

def ErrorFunctionDer(name:str = 'MSE'):
    if name == 'MSE' : return lambda y,t : y-t#y-t
    if name == 'altMSE' : return lambda y,t : 2*(y-t)
    if name == 'Xentropy' : return delta_cross_entropy

def cross_entropy_2(X,y):
    loss = -np.sum(y*np.log(X)+(1-y)*np.log(1-X))
    return loss
#LogLiky Loss
def delta_cross_entropy(X,y): # dL -> dL@ds -> 1 
    X[(y==1)] -= 1
    return X

def fit(X :np.array,y:np.array,lr,loss,tX=[],ty=[]):
    network['lr'] = lr
    network['loss'] = ErrorFunction(loss)
    network['lossDer'] = ErrorFunctionDer(loss)
    X,y = shuffle_data(X,y)
    # the Algorithm
    print("inital prediction")
    print("Train")
    pred(X,y)
    if not(len(tX) == 0 or len(ty) == 0):
        print("Test")
        pred(tX,ty)
    for epoch in range(200):
        loss = []
        print(f"epoch {epoch}")
        for xi,yi in zip(X,y):
            outputs,sum = forwardPropgation(xi)
            losscalc(loss,outputs,yi)             
            backwardProgration(outputs,sum,yi)
            update(network['lr'])
        print(f"train_loss: {np.mean(loss)}")
        if not(len(tX) == 0 or len(ty) == 0):
            loss
            for xi,yi in zip(tX,ty):
                outputs,sum = forwardPropgation(xi)
                losscalc(loss,outputs,yi) 
            print(f"train_loss: {np.mean(loss)}") 
        print("Predictions")
        print("Train")
        pred(X,y)
        if not(len(tX) == 0 or len(ty) == 0):
            print("Test")
            pred(tX,ty)

    return 
def forwardPropgation(x):
    sum_input = [x]
    outputs = [x]
    for w,act_func in zip(network['weigth'],network['activition']):
        sum_input.append((w@sum_input[-1]))
        outputs.append(act_func(w@outputs[-1]))
    
    return outputs,sum_input

def backwardProgration(outputs,isum,true_label):
    errd = network['lossDer']
    dLoss = errd(outputs[-1].reshape(1,-1),true_label.reshape(1,-1))        #chanded dim                                            
    for i in range(len(outputs)-1,-1,-1):
        dActivation = network['activitionDer'][i-1]
        dfo = dActivation(isum[i])                                                 
        if i != 0:
            #TODO dLoss@ anpassen
            dLdf = (dLoss*dfo).reshape(-1,1) #if i != len(outputs)-1 else (dLoss@dfo).reshape(-1,1)
            network['dweigth'][i-1] = dLdf@outputs[i-1].reshape(1,-1)                      
            dLoss = dLdf.reshape(1,-1)@network['weigth'][i-1]                                      
    return
def update(lr):
    for i in range(len(network['weigth'])):
        network['weigth'][i] -= lr*network['dweigth'][i]
def losscalc(loss,output,true_label):
     err = network['loss']
     loss.append(np.sum(err(output[-1],true_label)))
###
### END IMPLEMENTATIONS
###
X_train,y_train,X_test,y_test = loadData(PATH)
y_train = OneHotEncoding(y_train)
y_test = OneHotEncoding(y_test)
# hyperparameter 
# TODO bias koeffezient ?
neuron_net_layout = [X_train.shape[1],20,10,10]
neuron_net_layout = [X_train.shape[1],10]
neuron_net_layout = [X_train.shape[1],100,100,10]
neuron_net_act_f = ['ReLU','ReLU','softmax'] 
#neuron_net_act_f = ['softmax'] 
### WTF softmax 10,10  90%
BuildNN(neuron_net_layout,neuron_net_act_f)
fit(X_train,y_train,0.01,'Xentropy',X_test,y_test)       # 10 ,10 , 10 ReLU ReLU ReLU 0.0001 ## 100,50 
# 0.01 10,10 RELU SOFTMAX  MSE 64% acc 
# 0.01 100,10 RELU SOFTMAX XE -> 99 - 100 T 88-89
#########################################################################################
# funfact see abow is maybe ML not XE
#X = np.array([0.55,0.02,0.01,0.03,0.01,0.05,0.17,0.01,0.06,0.09])
#y = np.array([1,0,0,0,0,0,0,0,0,0])
#print(cross_entropy(X,y))
def fit_easy(hidden_act_lst,X,y):
    network['dweigth'] = network['weigth'].copy()
    network['activition'] = [ActivationFunc(name) for name in hidden_act_lst]
    network['activitionDer'] = [ActivationFuncDer(name) for name in hidden_act_lst]
    for xi,yi in zip(X,y):
        outputs,sum = forwardPropgation(xi)
        backwardProgration(outputs,sum,yi)
        update(network['lr'])
def oneoutlayer():
    network['weigth'] = [
                        np.array([[.1,.8],[.4,.6]]),
                        np.array([[.3,.9]])
                    ]
    X = np.array([[0.35,0.9]])
    y = np.array([[.5]])
    network['loss'] = ErrorFunction('MSE')
    network['lossDer'] = ErrorFunctionDer('MSE')
    network['lr'] = 0.5
    hidden_act_lst = ['ReLU','ReLU']
    fit_easy(hidden_act_lst,X,y)
def twooutlayer():
    network['weigth'] = [
                            np.array([[.3,.7],[-.1,-.4],[.2,-.6]]),
                            np.array([[-.3,-.8,.4],[.2,.2,.6]])
                        ]
    X = np.array([[1,-1]])
    y = np.array([[-.2,.5]])
    network['loss'] = ErrorFunction('altMSE')
    network['lossDer'] = ErrorFunctionDer('altMSE')
    network['lr'] = 0.1
    hidden_act_lst = ['ReLU','ReLU']
    fit_easy(hidden_act_lst,X,y)
#print("layer 1 \n\n")
#oneoutlayer()
#print("layer 2 \n\n")
#twooutlayer()