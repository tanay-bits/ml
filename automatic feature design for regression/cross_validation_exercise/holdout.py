from pylab import *

K = 6

def k_fold_cross_validation(data, K): 
    trains = [0]*K
    tests = [0]*K

    for k in xrange(K):
        trains[k] = asarray([x for i, x in enumerate(data) if i % K != k])
        tests[k] = asarray([x for i, x in enumerate(data) if i % K == k])

    return trains, tests


def load_data():
    from numpy import random

    data = matrix(genfromtxt('galileo_ramp_data.csv', delimiter=','))
    random.shuffle(data)
    data = asarray(data)

    data_train = (k_fold_cross_validation(data, K))[0]
    data_test = (k_fold_cross_validation(data, K))[1]

    xfull = data[:,0]
    xfull.shape = (size(xfull),1)
    yfull = data[:,1]
    yfull.shape = (size(yfull),1)

    xtrains = []
    ytrains = []
    xtests = []
    ytests = []
    for i in range(K):       
        xtrains.append(asarray(data_train)[i][:,0])
        ytrains.append(asarray(data_train)[i][:,1])
        xtests.append(asarray(data_test)[i][:,0])
        ytests.append(asarray(data_test)[i][:,1])
    
    return xfull, yfull, xtrains, ytrains, xtests, ytests



# plot the data
def plot_data(x,y,deg):
    f,axs = plt.subplots(2,3,facecolor = 'white')
    for i in range(0,3):
        axs[0,i].scatter(x,y, s = 30,color = '0')
        axs[0,i].set_xlabel('$x$',fontsize=20,labelpad = -2)
        axs[0,i].set_ylabel('$y$',fontsize=20,rotation = 0,labelpad = 0)
        axs[0,i].set_xlim(0,7.5)
        axs[0,i].set_ylim(0,1)
        axs[0,i].set_aspect(8)
        s = 'D = ' + str(deg[i])
        axs[0,i].set_title(s,fontsize=15)
        axs[0,i].xaxis.set_ticks(arange(0,7.6))
        axs[0,i].yaxis.set_ticks(arange(0,1.1))

        axs[1,i].scatter(x,y, s = 30,color = '0')
        axs[1,i].set(aspect = 'equal')
        axs[1,i].set_xlabel('$x$',fontsize=20,labelpad = -4)
        axs[1,i].set_ylabel('$y$',fontsize=20,rotation = 0,labelpad = 0)
        axs[1,i].set_xlim(0,7.5)
        axs[1,i].set_ylim(0,1)
        axs[1,i].set_aspect(8)
        s = 'D = ' + str(deg[i + 3])
        axs[1,i].set_title(s,fontsize=15)
        axs[1,i].xaxis.set_ticks(arange(0,7.6))
        axs[1,i].yaxis.set_ticks(arange(0,1.1))

    return(axs)

# plot learned model
def plot_model(w,D,ax):
    s = asarray(linspace(0,7.2,100))
    s.shape = (size(s),1)
    f = []
    for k in range(1,D+1):
        f.append(w[k]*s**k)
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
def poly_features(x,D):
    g = lambda i,j: i**j
    F = g(x, range(D+1)).T
    return(F)



def main():
    
    # number of polynomial features to use
    deg = array([1,2,3,4,5,6])

    # load and plot the data
    xfull, yfull, xs, ys, xts, yts = load_data()

    mse_avs = zeros((6,6))
    mse_t_avs = zeros((6,6))

    for i in range(K):
        x = xs[i].reshape(len(yfull)-len(yfull)/K, 1)
        y = ys[i].reshape(len(yfull)-len(yfull)/K, 1)
        xt = xts[i].reshape(len(yfull)/K, 1)
        yt = yts[i].reshape(len(yfull)/K, 1)
        
        # axs = plot_data(x,y,deg)

        # generate polynomial features and fit to data
        mses = []
        mses_t = []
        for D_ind in range(0,6):
            # generate fourier feature transformation
            F = poly_features(x,deg[D_ind])
            Ft = poly_features(xt,deg[D_ind])

            # get weights
            w = dot(linalg.pinv(dot(F,F.T)),dot(F,y))

            # compute mean squared error with current model
            new_mse = linalg.norm(dot(F.T,w) - y)/size(y)
            mses.append(new_mse)

            new_mse_t = linalg.norm(dot(Ft.T,w) - yt)/size(yt)
            mses_t.append(new_mse_t)

        mse_avs[i] = asarray(mses)
        mse_t_avs[i] = asarray(mses_t)

    # print mse_avs
    # print mse_t_avs

    mse_avf = mean(mse_avs, axis=0)
    mse_t_avf = mean(mse_t_avs, axis=0)
    # print mse_avf
    # print mse_t_avf

    plt.plot(range(1,7), mse_avf, '--', label='training avg error', linewidth=2)
    plt.plot(range(1,7), mse_t_avf, label='testing avg error', linewidth=2)
    xlabel('D')
    ylabel('Avg MSE over K=6 folds') 
    legend()

    D_best = argmin(mse_t_avf) + 1
    # print D_best
    Fbest = poly_features(xfull, D_best)
    wbest = dot(linalg.pinv(dot(Fbest,Fbest.T)),dot(Fbest,yfull))
    # mse_best = linalg.norm(dot(Fbest.T,wbest) - yfull)/size(yfull)
    # print 'MSE with the best model is = ', mse_best

    f,ax = plt.subplots(1,facecolor = 'white')

    ax.scatter(xfull,yfull, s = 30,color = '0')
    ax.set_xlabel('$x$',fontsize=20,labelpad = -2)
    ax.set_ylabel('$y$',fontsize=20,rotation = 0,labelpad = 0)
    # ax.set_xlim(0,1)
    # ax.set_ylim(-2,5)
    # ax.set_aspect(0.1)
    s = 'D_best = ' + str(D_best)
    ax.set_title(s,fontsize=15)
    # ax.xaxis.set_ticks(arange(0,1.1))
    # ax.yaxis.set_ticks(arange(-2,5.1))

    plot_model(wbest,D_best,ax)


    show()

        


    # # plot mean squared error for each degree tried
    # plot_mses(mses,mses_t,deg)

    # D_best = argmin(mses_t) + 1

    # Fbest = fourier_features(xfull, D_best)
    # wbest = dot(linalg.pinv(dot(Fbest,Fbest.T)),dot(Fbest,yfull))
    # mse_best = linalg.norm(dot(Fbest.T,wbest) - yfull)/size(yfull)
    # print 'MSE with the best model is = ', mse_best

    # f,ax = plt.subplots(1,facecolor = 'white')

    # ax.scatter(xfull,yfull, s = 30,color = '0')
    # ax.set_xlabel('$x$',fontsize=20,labelpad = -2)
    # ax.set_ylabel('$y$',fontsize=20,rotation = 0,labelpad = 0)
    # ax.set_xlim(0,1)
    # ax.set_ylim(-2,5)
    # ax.set_aspect(0.1)
    # s = 'D_best = ' + str(D_best)
    # ax.set_title(s,fontsize=15)
    # ax.xaxis.set_ticks(arange(0,1.1))
    # ax.yaxis.set_ticks(arange(-2,5.1))

    # plot_model(wbest,D_best,ax)
    # plot_dgf(ax)

    # legend()
    # show()

main()