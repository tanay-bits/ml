Basic Numerical Optimization
=======================================
*grad_descent.py* implements gradient descent for the simple convex cost function g(**w**) = **w**<sup>T</sup>**w** where **w** is a 10-dimensional vector input, and compares the effect of three different fixed step lengths. The initial point is w<sup>0</sup> = 10<sup>.</sup>**1**<sub>Nx1 </sub> and the stopping criterion is 100 iterations.

![gd](https://raw.githubusercontent.com/tanay-bits/ml/newyear/basic%20numerical%20optimization/gd.png)

*newtons_method.py* performs Newton’s method to find the minimum of the function g(**w**) = log(1+*e*<sup>**w**<sup>T</sup>**w**</sup>).

![nm](https://raw.githubusercontent.com/tanay-bits/ml/newyear/basic%20numerical%20optimization/nm.png)

Even though the cost functions are not the same in the above cases, in general Newton's method converges much faster (for convex cost functions; regularizers should be used for non-convex functions) than gradient descent. 