# 2. Basics of NN Programming:

- Binary Classification: categorize input data into one of two distinct classes.
Example: if we want to classify a cat/non-cat, a NN would process image data, output layer produces a probability indicating likelihood of a cat which can be classified by a threshold value.

- Logistic Regression: Given x, ŷ = P(y = 1 | x); 0 ≤ y ≤ 1. x ∈ R^nx. Parameters: w ∈ R^n; b ∈ R.
Output: ŷ = σ(w^Tx + b). σ is sigmoid function. (this is a logistic-regression algorithm)
σ(z) = 1/1 + e^-z. If z is large then σ(z) = 1 vice-versa, if z is large negative number then, σ(z)=0.
Effectively a single neuron designed for a binary classification, with a sigmoid activation function is mathematically equivalent to logistic regression.
y = true label (whether someone brought the product/not)
ŷ = predicted output of our model

- Logistic Regression Cost Function: 
A Loss Function measures the error/penalty of a single data point/instance.
Given a training set, we want ŷ^(i) = y^(i)
Loss (error) function: L(ŷ ,y) = - (y logŷ + (1-y) log (1 - ŷ)), instead of classic mean-square error since, it results in a non-convex cost function which makes it difficult to find the global minima.
If, y = 1: loss-function = -logŷ (large). If, y = 0: loss-function = -log (1-ŷ) (small).

- Cost Function J(w,b): Measures the efficiency of the entire training set, aggregate the losses over entire batch of data. Note: b is bias, w are weights. 
Cost-Function is average of all loss functions over the the entire training data (m).
Goal is to find appropriate w, b which minimize the cost-function.(gradient descent optimization)

- Gradient Descent: Find w, b that minimize J(w,b)
Repeat: w:= w - α. dJ(w)/dw, where α is the learning rate. (partial derivative for multi-variable)
learning rate is a hyperparameter which determines the step size taken during each iteration to minimize cost function.
Similar derivative applies to ‘b’, bias, as well.
In a nutshell, we measure the slope of a variable, getting a ‘w’ or ‘b’ value after which the function becomes relatively more convex after each iteration hence leading to a global minima.|
This minimizes the difference between network’s predictions and actual answers.
Note: Visualize gradient descent in NN and in general too.

- Computation Graph: 
It is  a flowchart which shows the steps needed to compute a function, each step in the graph represents a calculation or operation.

- Logistic Regression Gradient Descent: 
Recap of LR: Consider x1,x2: z = w1x1 + w2x2 + b; ŷ (a) = σ(z); hence, compute the loss.
By backpropagation method, da = -y/a + (1-y/1-a). dz = a-y. 
Subsequently, dw1, dw2 can be computer by partial derivative of loss-function wrt the weight.

- Gradient Descent on ‘m’ Examples (Training Data)