
from numpy import *
import matplotlib.pyplot as plt
import numdifftools as nd


def func(x):
    return log(1+exp(dot(x,x)))
    

###### ML Algorithm functions ######
def newtons_method(w0):
    w = w0
    g_path = []
    
    # w_path = []
    # w_path.append(w)
    g_path.append(log(1+exp(dot(w,w))))

    # start gradient descent loop
    grad = 1
    iter = 1
    max_its = 100
    while iter <= max_its:
        # take gradient step
        grad = gradient(g_path[-1])
        hess = nd.Hessian(func, n=w)
        w = w - linalg.pinv(hess)*grad

        # update path containers
        # w_path.append(w)
        g_path1.append(dot(w1,w1))
        g_path2.append(dot(w2,w2))
        g_path3.append(dot(w3,w3))
        iter+= 1
    

    # show final average gradient norm for sanity check
    # s = linalg.norm(grad)
    # s = 'The final average norm of the gradient = ' + str(float(s))
    # print(s)

    return (g_path)


def main():

    w0 = ones(10)
    
    g_path = newtons_method(w0)    # perform newton's method
    # print len(g_path)
    plt.plot(linspace(1,101,101), g_path1)
      
    # plt.legend(loc=4)
    plt.xlabel('iterations')
    plt.ylabel('objective value')
    plt.show()


main()