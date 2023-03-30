# mlpack-dag-network
[Proposal link](https://docs.google.com/document/d/1tznxFBU6UfgWLq4voV8eTg873s1JOpmCV5gZBQGAaaE/edit?usp=sharing)

DAG Network for mlpack

Working examples written using DAGNetwork class:
* ResNet_Example.cpp
* Thyroid_DAG.cpp 
* ResidualBlock2.cpp 
* SimpleTest.cpp

The example 1 and 2 also been written using the current mlpack API to compare results(See 'test' folder).

Example of the new Layer API:
1. split.cpp

You can compile these files with armadillo and mlpack to run.

### ResNet_Example.cpp

This is an ResNet18 trained on the subset(2000) of MNIST. It gives 79% accurarcy after two epochs.
Output of the program:

```
[INFO ] Loading 'data/digit-recognizer/train.csv' as CSV data.  Size is 785 x 42000.
Epoch 1/2
50/50 [====================================================================================================] 100% - 207.646s/epoch; 4152ms/step; loss: 9.25515
9.25515
Epoch 2/2
50/50 [====================================================================================================] 100% - 223.973s/epoch; 4479ms/step; loss: 2.15187
2.15187
Optimization Report
--------------------------------------------------------------------------------

Initial coordinates: 
 0.1085  0.6273 -0.0792  ... 0.0000

Final coordinates: 
 0.0772  0.6199  0.0628  ... 0.1295
iter          loss          loss change   |gradient|    step size     total time    
0             9.255         0.000         543.974       0.010         207.646       
1             2.152         7.103         207.087       0.010         431.619       

--------------------------------------------------------------------------------

Version:
ensmallen:                    2.18.2 (Fairmount Bagel)
armadillo:                    10.8.2 (Realm Raider)

Function:
Number of functions:          1600
Coordinates rows:             11180170
Coordinates columns:          1

Loss:
Initial                       9.255
Final                         2.152
Change                        7.103

Optimizer:
Maximum iterations:           3200
Reached maximum iterations:   false
Batch size:                   32
Iterations:                   2
Number of epochs:             3
Initial step size:            0.010
Final step size:              0.010
Coordinates max. norm:        543.974
Evaluate calls:               100
Gradient calls:               100
Time (in seconds):            431.619
Accuracy on Validation set: 79.800499
```

### Thyroid_DAG.cpp

In this example, I have reproduced the example from [ANN Tutorial](https://github.com/mlpack/mlpack/blob/master/doc/tutorials/ann.md) using DAG_Network.hpp.This program gives almost identical accuracy and output as the original program. 

Output of the program:


```
[INFO ] Loading 'data/thyroid_train.csv' as CSV data.  Size is 22 x 3772.
[INFO ] Loading 'data/thyroid_test.csv' as CSV data.  Size is 22 x 3428.
Optimization Report
--------------------------------------------------------------------------------

Initial coordinates: 
 0.5736 -0.4990  0.4213  ... 0.8353

Final coordinates: 
-0.0796 -0.1839 -0.1468  ... 0.6498
iter          loss          loss change   |gradient|    step size     total time    
0             0.315         0.000         3.497         0.010         0.002         
2             0.274         0.041         2.931         0.010         0.006         
4             0.237         0.037         3.205         0.010         0.010         
6             0.210         0.027         3.886         0.010         0.014         
8             0.194         0.015         4.563         0.010         0.018         
10            0.185         0.009         5.106         0.010         0.022         
12            0.179         0.006         5.515         0.010         0.026         
14            0.175         0.005         5.834         0.010         0.030         
16            0.171         0.004         6.099         0.010         0.034         
18            0.168         0.003         6.331         0.010         0.038         
20            0.164         0.003         6.539         0.010         0.042         
22            0.161         0.003         6.731         0.010         0.046         
24            0.158         0.003         6.912         0.010         0.050         

--------------------------------------------------------------------------------

Version:
ensmallen:                    2.18.2 (Fairmount Bagel)
armadillo:                    10.8.2 (Realm Raider)

Function:
Number of functions:          3772
Coordinates rows:             203
Coordinates columns:          1

Loss:
Initial                       0.315
Final                         0.156
Change                        0.159

Optimizer:
Maximum iterations:           100000
Reached maximum iterations:   false
Batch size:                   32
Iterations:                   26
Number of epochs:             27
Initial step size:            0.010
Final step size:              0.010
Coordinates max. norm:        6.999
Evaluate calls:               3129
Gradient calls:               3129
Time (in seconds):            0.052
Classification Error for the Test set: 0.056009
```
