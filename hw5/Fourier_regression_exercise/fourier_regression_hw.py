from pylab import *

# load the data
def load_data():
    # load data
    data = matrix(genfromtxt('noisy_sin_samples.csv', delimiter=','))
    x = asarray(data[:,0])
    x.shape = (size(x),1)
    y = asarray(data[:,1])
    y.shape = (size(y),1)
    return (x,y)

# plot the data
def plot_data(x,y,deg):
    f,axs = plt.subplots(2,3,facecolor = 'white')
    for i in range(0,3):
        axs[0,i].scatter(x,y, s = 30,color = '0')
        axs[0,i].set_xlabel('$x$',fontsize=20,labelpad = -2)
        axs[0,i].set_ylabel('$y$',fontsize=20,rotation = 0,labelpad = 0)
        axs[0,i].set_xlim(0,1)
        axs[0,i].set_ylim(-1.5,1.5)
        axs[0,i].set_aspect(0.333)
        s = 'D = ' + str(deg[i])
        axs[0,i].set_title(s,fontsize=15)
        axs[0,i].xaxis.set_ticks(arange(0,1.1))
        axs[0,i].yaxis.set_ticks(arange(-1,1.1))

        axs[1,i].scatter(x,y, s = 30,color = '0')
        axs[1,i].set(aspect = 'equal')
        axs[1,i].set_xlabel('$x$',fontsize=20,labelpad = -4)
        axs[1,i].set_ylabel('$y$',fontsize=20,rotation = 0,labelpad = 0)
        axs[1,i].set_xlim(0,1)
        axs[1,i].set_ylim(-1.5,1.5)
        axs[1,i].set_aspect(0.333)
        s = 'D = ' + str(deg[i + 3])
        axs[1,i].set_title(s,fontsize=15)
        axs[1,i].xaxis.set_ticks(arange(0,1.1))
        axs[1,i].yaxis.set_ticks(arange(-1,1.1))

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
def plot_mses(mses,deg):
    f,ax = plt.subplots(1,facecolor = 'white')
    ax.plot(deg,mses)
    ax.xaxis.set_ticks(deg)
    ax.set_xlabel('$D$',fontsize=20,labelpad = 10)
    ax.set_ylabel('$MSE $',fontsize=20,rotation = 0,labelpad = 20)

# generate fourier features
def fourier_features(x,D):
    ### YOUR CODE GOES HERE
    # g = lambda a,b: exp(1j*2*pi*b*a)
    # F1 = g(x, 0)
    # for i in range(1, D+1):
    #     f = g(x, i)
    #     F1 = hstack((F1,f))

    # return F1.T.real

    F = ones(x.shape)
    for m in range(1,D+1):
        f_i1 = cos(2*pi*m*x)
        f_i2 = sin(2*pi*m*x)           
        F = hstack((F,f_i1,f_i2))

    return F.T
        

def main():
    # 2*number of fourier features to use
    deg = array([1,3,5,7,9,15])

    # load and plot the data
    x,y = load_data()
    axs = plot_data(x,y,deg)

    # generate fourier features and fit to data
    mses = []
    for D in range(0,6):
        # generate fourier feature transformation
        F = fourier_features(x,deg[D])

        # get weights
        w = dot(linalg.pinv(dot(F,F.T)),dot(F,y))

        # compute mean squared error with current model
        new_mse = linalg.norm(dot(F.T,w) - y)/size(y)
        mses.append(new_mse)

        # plot fit to data
        n = 0
        m = D
        if D > 2:
            n = 1
            m = D - 3
        plot_model(w,deg[D],axs[n,m])

    # plot mean squared error for each degree tried
    plot_mses(mses,deg)

    show()

main()