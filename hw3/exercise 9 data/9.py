from numpy import *
import matplotlib.pyplot as plt

max_its = 10

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


def soft_counting_cost(X,y,w,P):
    count = 0

    for p in range(1, P+1):
        Xp = X[:,p-1]
        yp = y[p-1,0]
        choose_bw = array([0, sign(-yp*dot(Xp,w)[0])])
        maxp = amax(choose_bw)
        count = count + maxp

    return count




def sqm_grad_maker(X,y,w,P):
    g = zeros(w.shape)
    
    for p in range(1, P+1):
        Xp = X[:,p-1]
        yp = y[p-1,0]
        choose_bw = array([0, 1 - yp*dot(Xp,w)[0]])
        gp = amax(choose_bw)*yp*Xp
        gp.shape = w.shape
        g = g + gp

    g = -2*g
    return g


def sqm_hess_maker(X,y,w,N,P):
    h = zeros((N+1,N+1))

    for p in range(1, P+1):
        Xp = X[:,p-1]
        yp = y[p-1,0]

        if 1 - yp*dot(Xp,w)[0] > 0:
            Xp.shape = (N+1,1)
            hp = dot(Xp, Xp.T)
            h = h + hp

    h = 2*h
    return h


def sqm_counting_cost(X,y,w,P):
    count = 0

    for p in range(1, P+1):
        Xp = X[:,p-1]
        yp = y[p-1,0]
        choose_bw = array([0, sign(1 - yp*dot(Xp,w)[0])])
        maxp = amax(choose_bw)
        count = count + maxp

    return count



# Newton's method
def newtons_method(X,y,N,P):
    w_soft = zeros((N+1,1))
    w_sqm = array([[10],[0],[0],[0],[0],[0],[0],[0],[0]])
    
    misses_soft = []
    misses_sqm = []
    
    soft_grad = 1
    sqm_grad = 1
    
    k = 1

    
    while k <= max_its:
        
        soft_grad = soft_grad_maker(X,y,w_soft,P)
        soft_hess = soft_hess_maker(X,y,w_soft,N,P)
       
        w_soft = w_soft - dot(linalg.pinv(soft_hess),soft_grad)

        misses_soft_k = soft_counting_cost(X,y,w_soft,P)
        misses_soft.append(misses_soft_k)


        sqm_grad = sqm_grad_maker(X,y,w_sqm,P)
        sqm_hess = sqm_hess_maker(X,y,w_sqm,N,P)
        
        w_sqm = w_sqm - dot(linalg.pinv(sqm_hess),sqm_grad)

        misses_sqm_k = sqm_counting_cost(X,y,w_sqm,P)
        misses_sqm.append(misses_sqm_k)

        k += 1

    return w_soft, soft_grad, misses_soft, w_sqm, sqm_grad, misses_sqm



### main loop ###
def main():
    
    X,y,N,P = prep_data()

    # run Newton's method
    w_soft, soft_grad, misses_soft, w_sqm, sqm_grad, misses_sqm = newtons_method(X,y,N,P)

    
    print w_soft
    print soft_grad
    print misses_soft

    print w_sqm
    print sqm_grad
    print misses_sqm

    plt.plot(linspace(1,max_its,max_its), misses_soft)
    plt.plot(linspace(1,max_its,max_its), misses_sqm)
      
    # plt.legend(loc=4)
    plt.xlabel('iterations')
    plt.ylabel('number of missclassifications per iteration')
    # plt.axis([0,20,0,15])
    plt.show()

main()