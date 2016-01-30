Regression
=======================================
Transforming Nonlinear Models to Linear Models
----------

*sinfit.py* implements knowledge-driven design of **input feature** to convert the nonlinear sinusoidal relationship between x and y to a linear relationship between sin(x) and y. Now the least-squares cost function associated with this linear relationship can be easily solved for the global minimum by solving the system of linear equations obtained from setting the gradient equal to zero. Input vs. output as well as transformed input vs. output plots are shown below:

![sinfit](https://raw.githubusercontent.com/tanay-bits/ml/newyear/regression/3_5b1.png) ![sinfit2](https://raw.githubusercontent.com/tanay-bits/ml/newyear/regression/3_5b2.png)

In *ohmslaw.py*, the x ~ 1/y relationship warrants transforming the **output** (f(y) = 1/y) to form the linear system x ~ f(y).

![ohmslaw](https://raw.githubusercontent.com/tanay-bits/ml/newyear/regression/3_8_smooth.png) ![ohmslaw2](https://raw.githubusercontent.com/tanay-bits/ml/newyear/regression/3_8_transformed.png)	

In *linearlogistic.py*, nonlinear bacterial growth following a logistic sigmoid trend is transformed into a linear system by taking the **inverse** of the logistic function, and then solved for the optimal parameters using linear regression. That is,
σ(b+x<sub>p</sub>w) = y<sub>p</sub>
becomes
b+x<sub>p</sub>w =  log(y<sub>p</sub>/(1+y<sub>p</sub>)),  p = 1,..., P 

![lr](https://raw.githubusercontent.com/tanay-bits/ml/newyear/regression/3_10c.png)