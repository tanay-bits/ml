
from numpy import *
import matplotlib.pyplot as plt

###### ML Algorithm functions ######
def gradient_descent(w0, alpha1, alpha2, alpha3):
    w1 = w0
    w2 = w0
    w3 = w0
    g_path1 = []
    g_path2 = []
    g_path3 = []
    # w_path = []
    # w_path.append(w)
    g_path1.append(dot(w1,w1))
    g_path2.append(dot(w2,w2))
    g_path3.append(dot(w3,w3))

    # start gradient descent loop
    grad1 = 1
    grad2 = 1
    grad3 = 1
    iter = 1
    max_its = 100
    while iter <= max_its:
        # take gradient step
        grad1 = 2*w1
        grad2 = 2*w2
        grad3 = 2*w3
        w1 = w1 - alpha1*grad1
        w2 = w2 - alpha2*grad2
        w3 = w3 - alpha3*grad3

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

    return (g_path1, g_path2, g_path3)


def main():

    alpha1 = 10**-3
    alpha2 = 10**-1
    alpha3 = 1.001
    w0 = 10*ones(10)
    
    g_path1, g_path2, g_path3 = gradient_descent(w0, alpha1, alpha2, alpha3)    # perform gradient descent
    # print len(g_path)
    plt.plot(linspace(1,101,101), g_path1, label='alpha1=0.001')
    plt.plot(linspace(1,101,101), g_path2, label='alpha2=0.1')
    plt.plot(linspace(1,101,101), g_path3, label='alpha1=1.001')   
    plt.legend(loc=4)
    plt.xlabel('iterations')
    plt.ylabel('objective value')
    plt.show()


main()