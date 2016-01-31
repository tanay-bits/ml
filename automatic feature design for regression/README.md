Automatic Feature Design for Regression
============

Polynomial basis features for scalar input regression
-----------------------------------------------------
*noisy\_sin_samples.csv* contains noisy sinusoidal data as input, which is transformed using various degree *D* polynomial basis features
by *poly_regression.py*.

![altxt](https://raw.githubusercontent.com/tanay-bits/ml/newyear/automatic%20feature%20design%20for%20regression/3_2.png)

![altxt](https://raw.githubusercontent.com/tanay-bits/ml/newyear/automatic%20feature%20design%20for%20regression/3_1.png)

As is evident, despite the MSE (Mean Squared Error) decreasing as we increase the degree (more accurate fit to noisy data), D > 3 polynomial basis features *overfit* the data. D = 3 gives an accurate fit to the underlying *data-generating function*.

Fourier basis features for scalar input regression
--------------------------------------------------
Performed on the same dataset as above, but now using various degree *D* Fourier basis features (for each *D* there are *2D* number of basis elements, a *cos(2πmx)* and a *sin(2πmx)* for *m = 1, ..., D*).

![altxt](https://raw.githubusercontent.com/tanay-bits/ml/newyear/automatic%20feature%20design%20for%20regression/4_1.png)

![altxt](https://raw.githubusercontent.com/tanay-bits/ml/newyear/automatic%20feature%20design%20for%20regression/4_2.png)

D = 1 provides the best representation of the underlying data-generating function (since it is a sinusoid).

Neural Network basis features for scalar input regression
--------------------------------------------------
*tanh\_regression_hw.py* explores how various initializations affect the result of an M = 4 single hidden layer neural network basis features fit to the noisy sinusoidal dataset. The gradient descent module is:

[b,**w**,**c**,**v**,obj\_val] = tanh_grad_descent(**x**,**y**,*i*)

Here x and y is the input and output data respectively, i is a counter that loads an initialization for all variables, and b, w, c, and v are the optimal variables learned via gradient descent. After executing the optimization three times using three different initializations, we get the following results, where the three fits are on the left and the objective value vs. number of iterations plot on the right:

![altxt](https://raw.githubusercontent.com/tanay-bits/ml/newyear/automatic%20feature%20design%20for%20regression/5.png)

