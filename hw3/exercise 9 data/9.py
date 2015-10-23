from numpy import *
import matplotlib.pyplot as plt

# prepare the data
def prep_data():
    
    data = genfromtxt('breast_cancer_data.csv', delimiter=',')
    xs = data[:,:-1]
    y = data[:,-1]
    P = xs.shape[0]
    N = xs.shape[1]
    y.shape = (P,1)
    unos = ones((P,1))
    X = concatenate((unos.T,xs.T),0)

    return (X,y,N,P)


###### ML Algorithm functions ######
def sigmoid(t):
    return 1/(1+exp(-t))


def soft_grad_maker(X,y,w,P):
    g = zeros(w.shape)
    
    # t = -y*dot(X.T,w)
    # r = sigmoid(t)
    # z = y*r
    # onevec = ones(r.shape)
    
    # g = dot(dot(X, diagflat(z)), onevec)
    # g = dot(X,z)
    
    for p in range(1, P+1):
        gp = -sigmoid(-y[p-1,0]*dot(X[:,p-1],w)[0])*y[p-1,0]*X[:,p-1]
        gp.shape = w.shape
        g = g + gp

    return g


def soft_hess_maker(X,y,w,N,P):
    h = zeros((N+1,N+1))
    
    for p in range(1, P+1):
        t = -y[p-1,0]*dot(X[:,p-1],w)[0]
        Xp = X[:,p-1]
        Xp.shape = (N+1,1)
        hp = sigmoid(t)*(1 - sigmoid(t))*dot(Xp, Xp.T)
        h = h + hp

    return h


def counting_cost(X,y,w,P):
    count = 0

    for p in range(1, P+1):
        Xp = X[:,p-1]
        yp = y[p-1,0]
        choose_bw = array([0, sign(-yp*dot(Xp,w)[0])])
        maxp = amax(choose_bw)
        count = count + maxp

    return count




# run Newton's method
def newtons_method(X,y,N,P):
    w = zeros((N+1,1))
    misses = []
    grad = 1
    k = 1
    max_its = 10
    while k <= max_its:
        
        grad = soft_grad_maker(X,y,w,P)
        hess = soft_hess_maker(X,y,w,N,P)
       
        w = w - dot(linalg.pinv(hess),grad)

        misses_k = counting_cost(X,y,w,P)
        misses.append(misses_k)

        k += 1

    return w, grad, misses



### main loop ###
def main():
    
    X,y,N,P = prep_data()

    # run gradient descent
    w, grad, misses = newtons_method(X,y,N,P)

    # plot everything
    print w
    print grad
    print misses

    plt.plot(linspace(1,10,10), misses)
      
    # plt.legend(loc=4)
    plt.xlabel('iterations')
    plt.ylabel('number of missclassifications per iteration')
    # plt.axis([0,20,0,15])
    plt.show()

main()