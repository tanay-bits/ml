from pylab import *

# load the data
def load_data():
    # load data
    data = matrix(genfromtxt('4class_data.csv', delimiter=','))
    x = asarray(data[:,0:2])
    temp = shape(x)
    temp = ones((temp[0],1))
    X = concatenate((temp,x),1)
    X = X.T
    y = asarray(data[:,2])
    y.shape = (size(y),1)

    # return needed variables
    return (X,y)

###### ML Algorithm functions ######
# learn all C separators
def learn_separators(X,y):
    # YOUR CODE GOES HERE
    w0 = zeros((len(X),1))
    W = list()
    
    for c in range(1,5):
        yc = zeros((size(y),1))
        for p in range(size(y)):
            if y[p,0] == c:
                yc[p,0] = 1
            else:
                yc[p,0] = -1
        wc = newtons_method(w0,X,yc)
        W.append(wc.flatten())

    W = asarray(W)
    W = W.T
    return W

# run newton's method
def newtons_method(w0,X,y):
    w = w0

    # start newton's method loop
    H = dot(diag(y[:,0]),X.T)
    s = shape(y)
    s = s[0]
    l = ones((s,1))
    grad = 1
    k = 1
    max_its = 200
    while linalg.norm(grad) > 10**(-5) and k <= max_its:
        # compute gradient
        temp = 1/(1 + my_exp(dot(H,w)))
        grad = - dot(H.T,temp)

        # compute Hessian
        g = temp*(l - temp)
        hess = dot(dot(X,diag(g[:,0])),X.T)

        # take Newton step = solve Newton system
        temp = dot(hess,w) - grad
        w = dot(linalg.pinv(hess),temp)
        k += 1

    return w

# avoid overflow when using exp - just cutoff after arguments get too large/small
def my_exp(u):
    s = argwhere(u > 100)
    t = argwhere(u < -100)
    u[s] = 0
    u[t] = 0
    u = exp(u)
    u[t] = 1
    return u

###### plotting functions #######
# plot data and subproblem separators
def plot_data_and_subproblem_separators(X,y,W):
    # initialize figure, plot data, and dress up panels with axes labels etc.,
    num_classes = size(unique(y))
    color_opts = array([[1,0,0.4], [ 0, 0.4, 1],[0, 1, 0.5],[1, 0.7, 0.5],[0.7, 0.6, 0.5]])
    f,axs = plt.subplots(1,num_classes + 1,facecolor = 'white')

    r = linspace(0,1,150)
    for a in range(0,num_classes):
        # color current class
        axs[a].scatter(X[1,],X[2,], s = 30,color = '0.75')
        s = argwhere(y == a+1)
        s = s[:,0]
        axs[a].scatter(X[1,s],X[2,s], s = 30,color = color_opts[a,:])
        axs[num_classes].scatter(X[1,s],X[2,s], s = 30,color = color_opts[a,:])

        # draw subproblem separator
        z = -W[0,a]/W[2,a] - W[1,a]/W[2,a]*r
        axs[a].plot(r,z,'-k',linewidth = 2,color = color_opts[a,:])

        # dress panel correctly
        axs[a].set_xlim(0,1)
        axs[a].set_ylim(0,1)
        axs[a].set(aspect = 'equal')
    axs[num_classes].set(aspect = 'equal')

    return axs

# fuse individual subproblem separators into one joint rule
def plot_joint_separator(W,axs,num_classes):
    r = linspace(0,1,300)
    s,t = meshgrid(r,r)
    s = reshape(s,(size(s),1))
    t = reshape(t,(size(t),1))
    h = concatenate((ones((size(s),1)),s,t),1)
    f = dot(W.T,h.T)
    z = argmax(f,0)
    f.shape = (size(f),1)
    s.shape = (size(r),size(r))
    t.shape = (size(r),size(r))
    z.shape = (size(r),size(r))

    for i in range(0,num_classes + 1):
        axs[num_classes].contour(s,t,z,(i + 0.5,i + 0.5),colors = 'k',linewidths = 2.25)

def main():
    # load the data
    X,y = load_data()

    # learn all C vs notC separators
    W = learn_separators(X,y)

    # plot data and each subproblem 2-class separator
    axs = plot_data_and_subproblem_separators(X,y,W)

    # plot fused separator
    plot_joint_separator(W,axs,size(unique(y)))

    show()

main()