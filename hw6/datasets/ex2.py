from pylab import *

def k_fold_cross_validation(data, K): 
    trains = [0]*K
    tests = [0]*K

    for k in xrange(K):
        trains[k] = asarray([x for i, x in enumerate(data) if i % K != k])
        tests[k] = asarray([x for i, x in enumerate(data) if i % K == k])

    return trains[0], tests[0]


def load_data():
    from numpy import random

    data = matrix(genfromtxt('wavy_data.csv', delimiter=','))
    random.shuffle(data)
    data = asarray(data)
    
    K = 3

    data_train = (k_fold_cross_validation(data, K))[0]
    data_test = (k_fold_cross_validation(data, K))[1]

    xfull = data[:,0]
    xfull.shape = (size(xfull),1)
    yfull = data[:,1]
    yfull.shape = (size(yfull),1)

    xtrain = data_train[:,0]
    xtrain.shape = (size(xtrain),1)
    ytrain = data_train[:,1]
    ytrain.shape = (size(ytrain),1)

    xtest = data_test[:,0]
    xtest.shape = (size(xtest),1)
    ytest = data_test[:,1]
    ytest.shape = (size(ytest),1)
    
    return xfull, yfull, xtrain, ytrain, xtest, ytest



# plot the data
def plot_data(x,y,deg):
    f,axs = plt.subplots(2,4,facecolor = 'white')
    for i in range(0,4):
        axs[0,i].scatter(x,y, s = 30,color = '0')
        axs[0,i].set_xlabel('$x$',fontsize=20,labelpad = -2)
        axs[0,i].set_ylabel('$y$',fontsize=20,rotation = 0,labelpad = 0)
        axs[0,i].set_xlim(0,1)
        axs[0,i].set_ylim(-2,5)
        axs[0,i].set_aspect(0.1)
        s = 'D = ' + str(deg[i])
        axs[0,i].set_title(s,fontsize=15)
        axs[0,i].xaxis.set_ticks(arange(0,1.1))
        axs[0,i].yaxis.set_ticks(arange(-2,5.1))

        axs[1,i].scatter(x,y, s = 30,color = '0')
        axs[1,i].set(aspect = 'equal')
        axs[1,i].set_xlabel('$x$',fontsize=20,labelpad = -4)
        axs[1,i].set_ylabel('$y$',fontsize=20,rotation = 0,labelpad = 0)
        axs[1,i].set_xlim(0,1)
        axs[1,i].set_ylim(-2,5)
        axs[1,i].set_aspect(0.1)
        s = 'D = ' + str(deg[i + 4])
        axs[1,i].set_title(s,fontsize=15)
        axs[1,i].xaxis.set_ticks(arange(0,1.1))
        axs[1,i].yaxis.set_ticks(arange(-2,5.1))

    return(axs)

# plot learned model
def plot_model(w,D,ax):
    s = asarray(linspace(0,1,100))
    s.shape = (size(s),1)
    f = []
    for m in range(1,D+1):
        f.append(w[2*m-1]*cos(2*pi*m*s))
        f.append(w[2*m]*sin(2*pi*m*s))
    f = asarray(f)
    f = sum(f,axis = 0) + w[0]
    ax.plot(s,f,'-r', linewidth = 2)

# plot mean squared error
def plot_mses(mses,mses_t,deg):
    f,ax = plt.subplots(1,facecolor = 'white')
    ax.plot(deg,mses,deg,mses_t)
    ax.xaxis.set_ticks(deg)
    ax.set_xlabel('$D$',fontsize=20,labelpad = 10)
    ax.set_ylabel('$MSE $',fontsize=20,rotation = 0,labelpad = 20)

# generate fourier features
def fourier_features(x,D):
    F = ones(x.shape)
    for m in range(1,D+1):
        f_i1 = cos(2*pi*m*x)
        f_i2 = sin(2*pi*m*x)           
        F = hstack((F,f_i1,f_i2))

    return F.T



def main():
    
    # number of fourier features to use/2
    deg = array([1,2,3,4,5,6,7,8])

    # load and plot the data
    xfull, yfull, x, y, xt, yt = load_data()
    axs = plot_data(x,y,deg)

    # generate fourier features and fit to data
    mses = []
    mses_t = []
    for D_ind in range(0,8):
        # generate fourier feature transformation
        F = fourier_features(x,deg[D_ind])
        Ft = fourier_features(xt,deg[D_ind])

        # get weights
        w = dot(linalg.pinv(dot(F,F.T)),dot(F,y))

        # compute mean squared error with current model
        new_mse = linalg.norm(dot(F.T,w) - y)/size(y)
        mses.append(new_mse)

        new_mse_t = linalg.norm(dot(Ft.T,w) - yt)/size(yt)
        mses_t.append(new_mse_t)

        # plot fit to data
        n = 0
        m = D_ind
        if D_ind > 3:
            n = 1
            m = D_ind - 4
        plot_model(w,deg[D_ind],axs[n,m])

    # plot mean squared error for each degree tried
    plot_mses(mses,mses_t,deg)

    D_best = argmin(mses_t) + 1

    Fbest = fourier_features(xfull, D_best)
    wbest = dot(linalg.pinv(dot(Fbest,Fbest.T)),dot(Fbest,yfull))
    mse_best = linalg.norm(dot(Fbest.T,wbest) - yfull)/size(yfull)
    print 'MSE with the best model is = ', mse_best

    f,ax = plt.subplots(1,facecolor = 'white')

    ax.scatter(xfull,yfull, s = 30,color = '0')
    ax.set_xlabel('$x$',fontsize=20,labelpad = -2)
    ax.set_ylabel('$y$',fontsize=20,rotation = 0,labelpad = 0)
    ax.set_xlim(0,1)
    ax.set_ylim(-2,5)
    ax.set_aspect(0.1)
    s = 'D_best = ' + str(D_best)
    ax.set_title(s,fontsize=15)
    ax.xaxis.set_ticks(arange(0,1.1))
    ax.yaxis.set_ticks(arange(-2,5.1))

    plot_model(wbest,D_best,ax)

    show()

main()