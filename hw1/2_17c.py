
from numpy import *
import matplotlib.pyplot as plt
    

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
        grad = (2*exp(dot(w,w))/(1+exp(dot(w,w))))*w
        hess = (4*exp(dot(w,w))/(1+exp(dot(w,w)))**2)*outer(w,w) + (2*exp(dot(w,w))/(1+exp(dot(w,w))))*identity(10)
        # print hess

        w = w - dot(linalg.pinv(hess),grad)

        # update path containers
        # w_path.append(w)
        g_path.append(dot(w,w))
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
    plt.plot(linspace(1,101,101), g_path)
      
    # plt.legend(loc=4)
    plt.xlabel('iterations')
    plt.ylabel('objective value')
    plt.axis([0,20,0,15])
    plt.show()


main()