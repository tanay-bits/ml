from numpy import *
import matplotlib.pyplot as plt

# prepare the data
def prep_data():
    
    data = genfromtxt('spambase_data.csv', delimiter=',')
    xs = data[:,:-1]
    y = data[:,-1]
    P = xs.shape[0]
    # N = xs.shape[1]
    y.shape = (P,1)
    unos = ones((P,1))
    X = concatenate((unos.T,xs.T),0)
    X[-3:,:] = log10(X[-3:,:])
    X1 = X[:-9,:]
    X2 = X[:-3,:]

    return (X,X1,X2,y,P)


def grad_maker(X,y,w,P):
    g = -2*dot(dot(X, diagflat(y)), maximum(zeros((P,1)), ones((P,1)) - dot(diagflat(y), dot(X.T, w))))
    return g


def counting_cost(X,y,w,P):
    count = 0

    for p in range(1, P+1):
        Xp = X[:,p-1]
        yp = y[p-1,0]
        
        if sign(-yp*dot(Xp,w)[0]) > 0:
            count = count + 1

    return count


def gradient_descent(X,X1,X2,y,P):
    # w = random.randn(N+1,1)*10**(-3)
    w = zeros((len(X), 1))
    w1 = zeros((len(X1), 1))
    w2 = zeros((len(X2), 1))
    
    # containers for missclassifications per iteration:
    misses = []
    misses1 = []
    misses2 = []

    # grad = 1
    k = 1

    # tune max_its and alpha:
    max_its = 100
    alpha = 10**(-5)
    alpha1 = 10**(-5)
    alpha2 = 10**(-5)

    while k <= max_its:
        
        grad = grad_maker(X,y,w,P)
        grad1 = grad_maker(X1,y,w1,P)
        grad2 = grad_maker(X2,y,w2,P)
       
        w = w - alpha*grad
        w1 = w1 - alpha1*grad1
        w2 = w2 - alpha2*grad2

        misses_k = counting_cost(X,y,w,P)
        misses.append(misses_k)
        
        misses1_k = counting_cost(X1,y,w1,P)
        misses1.append(misses1_k)

        misses2_k = counting_cost(X2,y,w2,P)
        misses2.append(misses2_k)

        k += 1
        print misses1_k, misses2_k, misses_k
        

    return misses, misses1, misses2, k-1


### main loop ###
def main():
    # load and prepare the data:
    X,X1,X2,y,P = prep_data()
    
    # run gradient descent for squared margin and get the relevant parameters:
    misses, misses1, misses2, iters = gradient_descent(X,X1,X2,y,P)
        
    print 'missclassifications in final iter for BoW = ', misses1[-1]
    print 'missclassifications in final iter for BoW + char freq = ', misses2[-1]
    print 'missclassifications in final iter for BoW + char freq + spam features = ', misses[-1]

    # plots:
    plt.plot(linspace(1,iters,iters), misses1, label='BoW', lw=2.5)
    plt.plot(linspace(1,iters,iters), misses2, label='BoW + char freq', linestyle='--', lw=2.5)
    plt.plot(linspace(1,iters,iters), misses, label='BoW + char freq + spam features', linestyle=':', lw=2.5)

    plt.legend()  
    plt.xlabel('iterations')
    plt.ylabel('number of missclassifications per iteration')
    plt.show()

main()