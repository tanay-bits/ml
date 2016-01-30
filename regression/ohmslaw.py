from numpy import *
import matplotlib.pyplot as plt

def main():
    ohms_data = genfromtxt('ohms_data.csv', delimiter=',')

    xs = ohms_data[:,0]
    # fxs = 1/xs

    Xp = [e for e in xs]
    unos = [1 for e in xs]
    X = [unos, Xp]
    X = array(X)

    y = ohms_data[:,1]
    ytransformed = 1/y

    w_opt = dot(linalg.pinv(dot(X,X.T)), dot(X,ytransformed))
    b = w_opt[0]
    w = w_opt[1]
    print 'b_opt = ', b, '  w_opt = ', w
    
    # fitline = 1/(b + w*xs)
    fitline = 1/(b + w*linspace(0,105,106))

    plt.plot(xs, y, 'o')
    # plt.plot(xs, fitline)
    plt.plot(linspace(0,105,106), fitline)
    plt.ylim([0,5])
    plt.xlim([-1,105])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    plt.plot(xs, 1/y, 'o')
    plt.plot(linspace(0,105,106), 1/fitline)
    plt.xlabel('x')
    plt.ylabel('f(y)=1/y')
    plt.show()


main()