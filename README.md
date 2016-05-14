## Overview
###Realizing that A.I. produces A.I.
###Automation the evolution of neural network

###Details are explained in slideshare presentation below
<a href="http://www.slideshare.net/LauncherT/singularity-t-git">Automated Evolution of Neural Network</a>  

## Coding Level1
###I exlained coding in slideshare presentation below
<a href="http://www.slideshare.net/LauncherT/automated-evolution-level1">Automated Evolution of Neural Network coding explanation Level1</a>
###In this code, you can automate the evolution for image classification task of cifar10. i.e. the system can find optimized neural network architecture automatically.
###This is just a very simple demo of my thought, so the purpose is not achieving good accuracy score yet. Actually, the best accuracy was only 64% and that architecture was like below. It is because the system cannot split the input images or hidden inputs. In other words, all of the input image pixels is used in every cluster in next layer. So this system can not find architecture like convolution or inception so far.
clusters  
 [[248 392   0   0   0]  
 [116  34 355  35  14]  
 [246 296  68 214 474]  
 [292   0   0   0   0]  
 [ 85 122  99  15 205]  
 [ 14 289 227 278 190]  
 [  4   7 104 139   0]  
 [  0   0   0   0   0]]  
connections  
 [[[1 1 1 1 0]  
  [0 0 0 1 1]  
  [0 0 0 0 0]  
  [0 0 0 0 0]  
  [0 0 0 0 0]]  
  
 [[0 1 1 1 0]  
  [0 0 0 1 0]  
  [1 1 0 1 0]  
  [0 0 1 0 1]  
  [1 1 1 0 1]]  
  
 [[1 0 0 0 0]  
  [1 0 0 0 0]  
  [1 0 0 0 0]  
  [1 0 0 0 0]  
  [1 0 0 0 0]]  
  
 [[1 1 1 1 1]  
  [0 0 0 0 0]  
  [0 0 0 0 0]  
  [0 0 0 0 0]  
  [0 0 0 0 0]]  
  
 [[1 1 0 1 1]  
  [1 0 0 0 0]  
  [0 1 0 1 0]]  
  
 [[1 0 1 1 0]  
  [0 1 1 0 0]  
  [0 1 0 0 0]  
  [0 1 1 1 0]  
  [0 0 1 1 0]]  
  
 [[1 0 0 0 0]  
  [1 0 0 0 0]  
  [1 0 0 0 0]  
  [1 0 0 0 0]  
  [0 0 0 0 0]]  
  
 [[0 0 0 0 0]  
  [0 0 0 0 0]  
  [0 0 0 0 0]  
  [0 0 0 0 0]  
  [0 0 0 0 0]]]  

##Requirements
###Tensorflow
###Python  
###-my environment-  
Tensorflow 0.8.0(I found this codes do not work at 0.6.0)  
Ubuntu 14.04  

## Future work
###As I wrote above, splitting the input is important. But it is not innovative just to find the architecture which have already discovered before. On that point, I rather want to focus on automated evolution of time-series neural network to find unknown architecture for tasks such as Natural Language Processing.
