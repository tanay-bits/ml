from numpy import *
import matplotlib.pyplot as plt

def main():
    sin_data = genfromtxt('sinusoid_example_data.csv', delimiter=',')
    sorted_data = sin_data[sin_data[:,0].argsort()]

    xs = sorted_data[:,0]
    fxs = sin(2*pi*xs)

    Fp = [e for e in fxs]
    unos = [1 for e in fxs]
    F = [unos, Fp]
    F = array(F)

    y = sorted_data[:,1] 

    w_opt = dot(linalg.pinv(dot(F,F.T)), dot(F,y))
    b = w_opt[0]
    w = w_opt[1]
    print 'b_opt = ', b, 'w_opt = ', w
    
    fitline = b + w*sin(2*pi*linspace(0,1,50))

    plt.plot(xs, y, 'o')
    plt.plot(linspace(0,1,50), fitline)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    plt.plot(fxs, y, '*')
    plt.plot(sin(2*pi*linspace(0,1,50)), fitline)
    plt.xlabel('f(x)')
    plt.ylabel('y')
    plt.show()


main()