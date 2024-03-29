# Kaggle-Digit-Recognizer
The objective of the project was to estimate a tradeoff between the time and the accuracy of the algorithm that we use for multi class classification problem. We came up with few assumptions at the inception of the project. 

Assumption 1 was that higher accuracy will be achieved with more training time.

Assumption 2 was that the lesser the dimensions faster the training time. 

All these assumptions were based on the fact that our data was an image data in the form of a row-column dataset where each row represented an image and all 784 columns represented a pixel of value between white to black based on scale of 0 to 255.

However, we did observed that best algorithm with higher accuracy and shorter training time depends various hyper parameters. According to Figure 4, the SVM gives almost the same accuracy for kernel rbf with 75 principal components and kernel rbf with 50 principal components.
![Figure 4 ](https://github.com/taniyariar/Kaggle-Digit-Recognizer/blob/master/fig4.PNG)

Thus, our assumption 2 was justified that lesser dimensions with more variability run faster. As far as the deep learning approach is concerned for our problem we observed that 2D-CNN worked far better than Multi-Layer Perceptron for image processing. 
But our assumption 1 was disapproved as the NN with MLP accuracy reduced for higher training time. 
In conclusion, the whole crux is to find the suited algorithm for the problem and hyper tune the parameters to get an accuracy with reasonable training time as there always will be a tradeoff between the accuracy and time.
