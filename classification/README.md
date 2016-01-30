Classification
============
Binary Classification
-------------------------
*softmax_gd.py* performs 2-class classification by minimizing the **softmax** cost (which is a smooth approximation of the perceptron cost):

![sm](https://raw.githubusercontent.com/tanay-bits/ml/newyear/classification/softmax.png)

Instead of looping and summing over the entire dataset, which would be slow, a vectorized gradient descent is used. 

![4_3](https://raw.githubusercontent.com/tanay-bits/ml/newyear/classification/4_3c.png)

*breastcancer.py* compares the efficacy of the softmax and squared margin costs in distinguishing healthy from cancerous tissue using the entire breast cancer dataset as training data. This dataset consists of P = 569 datapoints, with each datapoint having nine medically valuable features (i.e., N = 9). The squared-margin cost minimization problem is:

![sm](https://raw.githubusercontent.com/tanay-bits/ml/newyear/classification/sqmargin.png)

This time optimization is performed via Newton's method, and number of misclassifications per iteration is also computed.

![sm](https://raw.githubusercontent.com/tanay-bits/ml/newyear/classification/4_9.png)

*facedetection.py* compares the efficacy of the softmax and squared margin costs in distinguishing face from non-face images using the face detection training dataset. This set of training data consists of P = 10,000 datapoints, 3,000 face images and 7,000 non-face images. Here each datapoint is a vectorized 28 x 28 grayscale image (i.e., N = 784).

![alt txt](https://raw.githubusercontent.com/tanay-bits/ml/newyear/classification/4_10long.png)

Since this time gradient descent is used, it takes way too many iterations to achieve reasonably accurate classification than it would have taken had we used Newton's method.

Multi-Class Classification
------------------------------
*one\_versus_all.py* performs One-vs-All classification on a 4-class dataset, i.e., finding the optimum weights for 4 two-class classification problems (using Newton's method on the softmax cost, in this case).

![alt txt](https://raw.githubusercontent.com/tanay-bits/ml/newyear/classification/ova4class.png)

*spamdetect.py* compares the efficacy of using various combinations of features to perform spam detection on a real dataset of emails, using the squared margin cost and gradient descent. The dataset consists of features taken from 1813 spam and 2788 real email messages (for a total of P = 4601 data-points), as the training data. The features for each data-point include: 48 BoW (Bag of Words) features, 6 character frequency features, and 3 spam-targeted features (further details on these features can be found by reviewing the readme file *spambase\_data_readme.txt*). This dataset may be found in *spambase_data.csv*. We can observe that using more types of features gives better results:

![alt txt](https://raw.githubusercontent.com/tanay-bits/ml/newyear/classification/4_19.png)

Training and Testing the MNIST Database of Handwritten Digits
-------------------------------------------------------------------
*MNIST.py* performs C = 10 multi-class classification for [handwritten digit recognition](https://en.wikipedia.org/wiki/MNIST_database), employing the One-vs-All multi-class classification framework. Softmax cost with Newton’s method is employed to solve each of the two-class subproblems.

+ Firstly, we train the classifier on the training set located in *MNIST\_training_data.csv*, that contains P = 60,000 examples of handwritten digits 0-9 (all examples are vectorized grayscale images of size 28 x 28 pixels). An accuracy of ~97% is obtained on this training data.
+ Using the weights learned from the training, our model showed ~92% accuracy on a new test dataset of handwritten digits located in *MNIST\_testing_data.csv*. This contains P = 10,000 new examples of handwritten digits that were not used in the training of our model.

