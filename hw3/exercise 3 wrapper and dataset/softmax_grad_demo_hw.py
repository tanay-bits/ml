from pylab import *

# load the data
def load_data():
    # load data
    data = matrix(genfromtxt('imbalanced_2class.csv', delimiter=','))
    X = asarray(data[:,0:2])
    y = asarray(data[:,2])
    y.shape = (size(y),1)
    return (X,y)


###### ML Algorithm functions ######
def sigmoid(t):
    return 1/(1+exp(-t))


def grad_summation(X,y,w):
    # g = zeros(w.shape)
    
    t = -y*dot(X.T,w)
    r = sigmoid(t)
    z = y*r
    # onevec = ones(r.shape)
    
    # g = dot(dot(X, diagflat(z)), onevec)
    g = -dot(X,z)
    
    # for p in range(1, size(y)+1):
    #     s = -sigmoid(-y[p-1,0]*dot(X[:,p-1],w)[0])*y[p-1,0]*X[:,p-1]
    #     s = array([[x] for x in s])
    #     g = g + s

    return g


# run gradient descent
def gradient_descent(X,y):
    # use compact notation and initialize
    temp = shape(X)
    temp = ones((temp[0],1))
    X = concatenate((temp,X),1)
    X = X.T
    w = randn(3,1)

    # start gradient descent loop
    grad = 1
    k = 1
    max_its = 3000
    alpha = 10**(-1)
    while linalg.norm(grad) > 10**(-5) and k <= max_its:
        # compute gradient
        grad = grad_summation(X,y,w)    # YOUR CODE GOES HERE

        # take gradient step
        w = w - alpha*grad

        # update path containers
        k += 1
        print(linalg.norm(grad))

    return w


###### plotting functions #######
def plot_all(X,y,w):

    # initialize figure, plot data, and dress up panels with axes labels etc.,
    fig = plt.figure(facecolor = 'white')
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('$x_1$',fontsize=20,labelpad = 20)
    ax1.set_ylabel('$x_2$',fontsize=20,rotation = 0,labelpad = 20)
    s = argwhere(y == 1)
    s = s[:,0]
    scatter(X[s,0],X[s,1], s = 30,color = (1, 0, 0.4))
    s = argwhere(y == -1)
    s = s[:,0]
    scatter(X[s,0],X[s,1],s = 30, color = (0, 0.4, 1))
    ax1.set_xlim(0,1.05)
    ax1.set_ylim(0,1.05)

    # plot separator
    r = linspace(0,1,150)
    z = -w.item(0)/w.item(2) - w.item(1)/w.item(2)*r
    ax1.plot(r,z,'-k',linewidth = 2)
    show()


### main loop ###
def main():
    # load data
    X,y = load_data()

    # run gradient descent
    w = gradient_descent(X,y)

    # plot everything
    plot_all(X,y,w)

main()