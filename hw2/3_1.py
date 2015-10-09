from numpy import *
import matplotlib.pyplot as plt

def main():
    student_debt = genfromtxt('student_debt.csv', delimiter=',')
    alpha = 0.01

    xs = student_debt[:,0]
    Xp = [xs[i] for i,e in enumerate(xs)]
    unos = [1 for i,e in enumerate(xs)]
    X = [unos, Xp]
    X = array(X)

    y = student_debt[:,1] 

    w_opt = dot(linalg.pinv(dot(X,X.T)), dot(X,y))
    b = w_opt[0]
    w = w_opt[1]
    fitline = b + w*xs

    prediction = b +w*2050
    print 'Total student debt in 2050 = ', prediction

    print w_opt

    plt.plot(xs, y, '*')
    plt.plot(xs, fitline)
    plt.xlabel('Year')
    plt.ylabel('Debt')
    plt.show()


main()