MATLAB code for finding influential examples using fastBoot distance metric as described in "Understanding Classifier Errors by Examining Influential Neighbors" by M. Kabra, A. Robie and K. Branson in CVPR 2015.

For the demo download mnist data set from http://www.cs.nyu.edu/~roweis/data/mnist_all.mat . Then execute InfluenceDemo.m in MATLAB.

If you get an error saying accummatrix function is missing, then you might have to compile accummatrix.cpp. Execute "mex accummatrix.cpp" on command line or in MATLAB.


