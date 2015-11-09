from pylab import *

###### basic data manipulation and plotting functions ######
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
def plot_data(x,y):
    f,axs = plt.subplots(1,2,facecolor = 'white')
    axs[0].scatter(x,y, s = 30,color = '0')
    axs[0].set_xlabel('$x$',fontsize=20,labelpad = -2)
    axs[0].set_ylabel('$y$',fontsize=20,rotation = 0,labelpad = 0)
    axs[0].set_xlim(0,1)
    axs[0].set_ylim(-1.5,1.5)
    axs[0].set_aspect(0.333)
    axs[0].xaxis.set_ticks(arange(0,1.1))
    axs[0].yaxis.set_ticks(arange(-1,1.1))
    return(axs)

# plot objective value at each iteration
def plot_obj(o,ax,color):
    if size(o) == 15000:
        ax[1].plot(o[100:])
        ax[1].xaxis.set_ticks([0,4900,9900,14900])
        labels = [item.get_text() for item in ax[1].get_xticklabels()]
        labels = array([100,5000,10000,15000])
        ax[1].set_xticklabels(labels)
    else:
        ax[1].plot(o)
    ax[1].set_xlabel('$k$',fontsize=20,labelpad = 0)
    ax[1].set_ylabel('$g(w^k)$',fontsize=20,rotation = 0,labelpad = 18)

# plot model onto data
def plot_approx(b,w,c,v,ax,color):
    M = size(c)
    s = linspace(0,1,100)
    s = array(s)
    s.shape = (size(s),1)
    t = b*ones((100,1))
    for m in range(0,M):
        t = t + w[m]*tanh(c[m] + v[m]*s)
    ax[0].plot(s,t)

###### ML functions ######
# gradient descent for single layer tanh nn basis
def tanh_grad_descent(x,y,i):
    # initialize weights etc.,
    b,w,c,v = initialize(i)
    P = size(x)
    M = 4            # number of neural network bases features to use
    alpha = 10**-3
    l_p = ones((P,1))

    # stopper and container
    max_its = 15000
    k = 1
    obj_val = []

    ### main loop
    while k <= max_its:
        # compute gradient
        ### YOUR CODE GOES HERE
        q = []
        t1 = []
        t2 = []
        t3 = []
        t4 = []
        s1 = []
        s2 = []
        s3 = []
        s4 = []

        for p in range(P):
            xp = x[p][0]
            yp = y[p][0]
            q.append(b + dot(tanh(c + xp*v).T, w) - yp)
            t1.append(tanh(c[0][0] + xp*v[0][0]))
            t2.append(tanh(c[1][0] + xp*v[1][0]))
            t3.append(tanh(c[2][0] + xp*v[2][0]))
            t4.append(tanh(c[3][0] + xp*v[3][0]))
            s1.append((cosh(c[0][0] + xp*v[0][0]))**-2)
            s2.append((cosh(c[1][0] + xp*v[1][0]))**-2)
            s3.append((cosh(c[2][0] + xp*v[2][0]))**-2)
            s4.append((cosh(c[3][0] + xp*v[3][0]))**-2)

        q = asarray(q).reshape(P,1)
        t1 = asarray(t1).reshape(P,1)
        t2 = asarray(t2).reshape(P,1)
        t3 = asarray(t3).reshape(P,1)
        t4 = asarray(t4).reshape(P,1)
        s1 = asarray(s1).reshape(P,1)
        s2 = asarray(s2).reshape(P,1)
        s3 = asarray(s3).reshape(P,1)
        s4 = asarray(s4).reshape(P,1)

        grad_b = (2*dot(l_p.T, q))[0][0]

        grad_w1 = 2*dot(l_p.T, q*t1)
        grad_w2 = 2*dot(l_p.T, q*t2)
        grad_w3 = 2*dot(l_p.T, q*t3)
        grad_w4 = 2*dot(l_p.T, q*t4)

        grad_w = vstack((grad_w1[0], grad_w2[0], grad_w3[0], grad_w4[0]))
        
        grad_c1 = 2*dot(l_p.T, q*s1)*w[0][0]
        grad_c2 = 2*dot(l_p.T, q*s2)*w[1][0]
        grad_c3 = 2*dot(l_p.T, q*s3)*w[2][0]
        grad_c4 = 2*dot(l_p.T, q*s4)*w[3][0]

        grad_c = vstack((grad_c1[0], grad_c2[0], grad_c3[0], grad_c4[0]))

        grad_v1 = 2*dot(l_p.T, q*x*s1)*w[0][0]
        grad_v2 = 2*dot(l_p.T, q*x*s2)*w[1][0]
        grad_v3 = 2*dot(l_p.T, q*x*s3)*w[2][0]
        grad_v4 = 2*dot(l_p.T, q*x*s4)*w[3][0]

        grad_v = vstack((grad_c1[0], grad_c2[0], grad_c3[0], grad_c4[0]))


        # take gradient steps
        b = b - alpha*grad_b
        w = w - alpha*grad_w
        c = c - alpha*grad_c
        v = v - alpha*grad_v

        # update stopper and container
        k = k + 1
        obj_val.append(calculate_obj_val(x,y,b,w,c,v))
    obj_val = array(obj_val)
    obj_val.shape = (size(obj_val),1)
    return(b,w,c,v,obj_val)

def calculate_obj_val(x,y,b,w,c,v):
    s = 0;
    P = size(x)
    for p in range(0,P):
        s = s + (b + dot(w.T,tanh(c + v*x[p])) - y[p])**2
    return(s)

def sech(z):
    s = 2/(exp(z) + exp(-z))
    return(s)

# random initializations
def initialize(i):
    b = 0
    w = 0
    c = 0
    v = 0
    if i == 1:
        b =array( [-0.4544])
        b.shape = (1,1)
        w = array([-0.3461,   -0.8727,   0.6312,   0.9760])
        w.shape = (4,1)
        c = array([ -0.6584,  0.7832,   -1.0260,   0.5559])
        c.shape = (4,1)
        v = array([-0.8571,  -0.8623,  1.0418,  -0.4081])
        v.shape = (4,1)

    elif i == 2:
        b =   array([-1.1724])
        b.shape = (1,1)
        w = array([  0.0950 ,  -1.9936,   -3.6876 ,  -0.6466])
        w.shape = (4,1)
        c = array([ -3.4814,  -0.3177,   -4.7905,   -1.5374])
        c.shape = (4,1)
        v = array([-0.7055,   -0.6778,   0.1639,   -2.4117])
        v.shape = (4,1)

    else:
        b =  array([0.1409])
        b.shape = (1,1)
        w = array([0.5207,   -2.1275,   10.7415,    3.5584])
        w.shape = (4,1)
        c = array([2.7754 ,   0.0417,   -5.5907,   -2.5756])
        c.shape = (4,1)
        v = array([-1.8030 ,  0.7578,   -2.4235,    0.6615])
        v.shape = (4,1)
    return(b,w,c,v)

def main():
    # load and plot the data
    x,y = load_data()
    axs = plot_data(x,y)

    # generate tanh features and fit to data/fit least squares cost to data
    colors = array(['r','g','b'])
    for i in range(0,3):
        b,w,c,v,obj_val = tanh_grad_descent(x,y,i)

        # plot model
        plot_approx(b,w,c,v,axs,colors[i])

        # plot objective value at each iteration of gradient descent
        plot_obj(obj_val,axs,colors[i])


    show()

main()