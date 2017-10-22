# Java implemention of miniFlow
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This project is created for learning purpose. The original implementation is python based provided by udacity, added here as well under python/ directory for reference.

This implemention doesn't rely on any third party libraries, it also includes "boston house price" [http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html] sample data embedded in the java file. No need to install any environment or tools, Just click run in Java IDE.

As java enforce the data type, so it is easier to read through and understand the data type of "values" or "gradients", which I found often confusing in python, as they are sometimes means matrix, and sometimes are vectors or scalars. Please also notice for "gradients" map, the key and value have same shape. 

In this implementation, the convention is upper case letter means matrix and lower case letter means vector or scalar. To enhance readability, I use "Value" class to represent any matrix and vector and scalar. Inside, Value is always a 2 dimensional double array, with row and col. For scalar, it is a "1 X 1" matrix, for vector, it is "1 X N" matrix. I found this is the clearest way to represent value and express the operation among them. I also use "dot" for matrix tranformation, while "multiple" mean multiple two elements from corresponding location. eg: [1, 2] dot Transpose([1, 2]) = [5], while [1, 2] * [1, 2] = [1, 4].  "A dot B" requires first matrix A number of colunms equals B number of rows. "A * B" requires A and B have same number of rows and columns. Please refer [https://github.com/maxiaodong97/miniFlow/blob/master/java/src/com/udacity/ama/miniFlow/Value.java] for details.


| Files                   | Description                                        |
| ----------------------- | -------------------------------------------------- |
| Graph.java	            | topological sort for neurons                       |
| Input.java	            | input node, eg, X, W, b                            |
| Linear.java	            | linear node to y = X dot W + b                     |    
| MSE.java	              | node to measure distance of two vector             |
| Sigmoid.java	          | node for activation function                       |
| Node.java               | base class of all nodes                            |
| Value.java	            | value represent for matrix, vector or scalar       | 
| test/BostonData.java	  | boston house price data 506 samples and 13 features|                      
| test/Main.java          | construct neural networks and performanc SGD       |

Here is brief flow for what main function does. 

First define the input values and the structure of the neural networks to predict the house price.
X_ is the data set of samples, it is a "506 X 13" matrix, row is the number of samples, and columns are 13 features, normalized.
y_ is the house price, it is "506 X 1" matrix, row is number of samples, column is the price. 
W1_, b1_, W2_, b2_ is the parameters we want to find out to best describe the data set using the neural network structure.
W1, b1, W2, b2 is the Input type of node. We pass optional value and a name for input type of node. Passing a user friendly name is easier for debugging. 

The neural networks structure is defined as, l1, s1, l2, and cost.
l1  = linear(X, W1, b1), since we define hidden layer nodes to 10, so l1 is a "10 X B" matrix. Where B is the batchSize.
s1  = sigmoid(l1), which is also a "10 X B" matrix.
l2  = linear(s1, W2, b2), it is "B X 1" matrix.
y is also "B X 1" matrix. 
cost = MSE(y, l2), the output is "1 X 1" matrix, measure the "sum of error" between y and l2.

``` java
    int numFeatures = BostonData.FEATURES;
    int numSamples = BostonData.SAMPLES;
    int numHidden = 10;

    Value X_ = new Value(BostonData.instance.DATA).normalizeColumn();
    Value y_ = new Value(BostonData.instance.TARGET).T();
    Value W1_ = new Value(numFeatures, numHidden).random();
    Value b1_ = new Value(1, numHidden).zero();
    Value W2_ = new Value(numHidden, 1).random();
    Value b2_ = new Value(1, 1).zero();
    Input X = new Input("X");
    Input y = new Input("Y");
    Input W1 = new Input(W1_, "W1");
    Input b1 = new Input(b1_, "b1");
    Input W2 = new Input(W2_, "W2");
    Input b2 = new Input(b2_, "b2");

    Node l1 = new Linear(X, W1, b1, "l1");
    Node s1 = new Sigmoid(l1, "s1");
    Node l2 = new Linear(s1, W2, b2, "l2");
    Node cost = new MSE(y, l2, "cost");
```

