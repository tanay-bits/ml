import numpy as np
import matplotlib.pyplot as plt

# load and prepare the data
def prep_data(filename):    
    data = np.genfromtxt(filename, delimiter=',')
    xs = data[:,:-1]
    y = data[:,-1]
    P = xs.shape[0]
    # N = xs.shape[1]
    y.shape = (P,1)
    unos = np.ones((P,1))
    X = np.concatenate((unos.T,xs.T),0)

    return X, y


###### ML Algorithm functions ######

# avoid overflow when using exp - just cutoff after arguments get too large/small
def my_exp(u):
    s = np.argwhere(u > 100)
    t = np.argwhere(u < -100)
    u[s] = 0
    u[t] = 0
    u = np.exp(u)
    u[t] = 1
    return u


# newton's method algo
def newtons_method(w0, X, y):
    w = w0
    grad = 1
    k = 1
    max_its = 20
    lam = 10**(-3)

    while k <= max_its:
        print k
        # compute gradient
        t = -y*np.dot(X.T,w)
        r = 1/(1+my_exp(-t))
        z = y*r
        grad = -np.dot(X,z)

        # compute l2 regularized Hessian
        hess = np.dot(X*r.flatten(), X.T) + lam*np.identity(len(X))
        # hess = np.dot(X*r.flatten(), X.T)

        # take Newton step = solve Newton system
        w = w - np.dot(np.linalg.pinv(hess), grad)

        k += 1

    return w.flatten()


# learn all C separators
def learn_separators(X, y):
    w0 = np.random.randn(len(X),1)*10**(-3)
    num_classes = np.size(np.unique(y))
    W_container = np.zeros((len(X),num_classes))
    
    for c in range(1, num_classes+1):
        yc = np.zeros((len(y),1))
        for p in range(len(y)):
            if y[p,0] == c:
                yc[p,0] = 1
            else:
                yc[p,0] = -1

        W_container[:,c-1] = newtons_method(w0,X,yc)
        
    return W_container


# calculate accuracy
def accuracy(X, y, W):
    count = 0
    for p in range(1, len(y)+1):
        Xp = X[:, p-1]
        yp = y[p-1, 0]
        Z = np.dot(Xp.T, W)
        yp_pred = np.argmax(Z) + 1

        if yp != yp_pred:
            count = count + 1    

    acc = 1 - count/float(len(y))
    return acc


###### main function ######
def main():
    # load the training data
    print 'Loading and preparing the training data...'
    training_file = 'MNIST_train_data.csv'
    X, y = prep_data(training_file)

    # learn all C vs notC separators
    print 'Learning the separators of training data...'
    W = learn_separators(X, y)
    
    # determine accuracy on training data
    accuracy_train = accuracy(X, y, W)

    print 'Loading and preparing the test data...'
    test_file = 'MNIST_test_data.csv'
    Xt, yt = prep_data(test_file)

    # learn all C vs notC separators
    print 'Learning the separators of test data...'
    Wt = learn_separators(Xt, yt)

    # determine accuracy on test data
    accuracy_test = accuracy(Xt, yt, Wt)

    print 'Accuracy on training data = ', accuracy_train
    print 'Accuracy on test data = ', accuracy_test


main()