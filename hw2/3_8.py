from numpy import *
import matplotlib.pyplot as plt

def main():
    ohms_data = genfromtxt('ohms_data.csv', delimiter=',')

    xs = ohms_data[1:,0]
    fxs = 1/xs

    Fp = [e for e in fxs]
    unos = [1 for e in fxs]
    F = [unos, Fp]
    F = array(F)

    y = ohms_data[1:,1] 

    w_opt = dot(linalg.pinv(dot(F,F.T)), dot(F,y))
    b = w_opt[0]
    w = w_opt[1]
    print 'b_opt = ', b, '  w_opt = ', w
    
    fitline = b + w*fxs

    plt.plot(xs, y, 'o')
    plt.plot(xs, fitline)
    plt.ylim([0,5])
    plt.xlim([0,105])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    plt.plot(fxs, y, 'o')
    plt.plot(fxs, fitline)
    plt.xlabel('f(x)=1/x')
    plt.ylabel('y')
    plt.show()


main()