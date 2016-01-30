###### Python code for exercise 4.10 ######
##   Tanay Choudhary | NetID: tcp867 ##

from numpy import *
import matplotlib.pyplot as plt

# prepare the data
def prep_data():
    
    data = genfromtxt('feat_face_data.csv', delimiter=',')
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

# Vectorized gradient for softmax cost function
def soft_grad_maker(X,y,w):
    
    t = -y*dot(X.T,w)
    r = sigmoid(t)
    z = y*r
    
    g = -dot(X,z)

    return g

# Gradient for squared margin perceptron
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

# Cost function to determine number of misclassifications
def counting_cost(X,y,w,P):
    count = 0

    for p in range(1, P+1):
        Xp = X[:,p-1]
        yp = y[p-1,0]
        
        if sign(-yp*dot(Xp,w)[0]) > 0:
            count = count + 1

    return count


# Gradient descent for the softmax cost
def gradient_descent_soft(X,y,N,P):
    w_soft = random.randn(N+1,1)*10**(-4)
    
    # container for missclassifications per iteration:
    misses_soft = []
    
    soft_grad = 1
    
    k = 1

    # tune max_its and alpha 
    max_its = 3000
    alpha = 0.01

    
    while k <= max_its:
        
        soft_grad = soft_grad_maker(X,y,w_soft)
       
        w_soft = w_soft - alpha*soft_grad

        misses_soft_k = counting_cost(X,y,w_soft,P)
        misses_soft.append(misses_soft_k)

        k += 1
        print linalg.norm(soft_grad), misses_soft_k

    return w_soft, soft_grad, misses_soft, k-1


# Gradient descent for the squared margin perceptron
def gradient_descent_sqm(X,y,N,P):
    w_sqm = random.randn(N+1,1)*10**(-3)
    
    # container for missclassifications per iteration:
    misses_sqm = []

    sqm_grad = 1

    k = 1

    # tune max_its and alpha:
    max_its = 3000
    alpha = 10**(-4)


    while k <= max_its:
        
        sqm_grad = sqm_grad_maker(X,y,w_sqm,P)
       
        w_sqm = w_sqm - alpha*sqm_grad

        misses_sqm_k = counting_cost(X,y,w_sqm,P)
        misses_sqm.append(misses_sqm_k)

        k += 1
        print linalg.norm(sqm_grad), misses_sqm_k
        

    return w_sqm, sqm_grad, misses_sqm, k-1



### main loop ###
def main():
    # load and prepare the data:
    X,y,N,P = prep_data()

    # run gradient descent for softmax and get the relevant parameters:
    w_soft, soft_grad, misses_soft, iters_soft = gradient_descent_soft(X,y,N,P)
    
    # run gradient descent for squared margin and get the relevant parameters:
    w_sqm, sqm_grad, misses_sqm, iters_sqm = gradient_descent_sqm(X,y,N,P)
        
    print 'missclassifications in final iter for softmax = ', misses_soft[-1]
    print 'missclassifications in final iter for squared margin = ', misses_sqm[-1]

    # plots:
    plt.plot(linspace(1,iters_soft,iters_soft), misses_soft, label='softmax')   
    plt.plot(linspace(1,iters_sqm,iters_sqm), misses_sqm, label='squared margin', linestyle='--')
      
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('number of missclassifications per iteration')
    plt.show()

main()