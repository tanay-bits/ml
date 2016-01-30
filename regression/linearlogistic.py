from numpy import *
import matplotlib.pyplot as plt

def main():
    bacteria_data = genfromtxt('bacteria_data.csv', delimiter=',')

    xs = bacteria_data[:,0]

    Xp = [e for e in xs]
    unos = [1 for e in xs]
    X = [unos, Xp]
    X = array(X)

    y = bacteria_data[:,1]

    z = log(y/(1-y)) 

    w_opt = dot(linalg.pinv(dot(X,X.T)), dot(X,z))
    b = w_opt[0]
    w = w_opt[1]
    print 'b_opt = ', b, 'w_opt = ', w
    
    fitline = 1/(1+exp(-b-w*linspace(0,24,25)))

    plt.plot(xs, y, 'o')
    plt.plot(linspace(0,24,25), fitline)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


main()