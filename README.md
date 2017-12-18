# Understanding-MLP-using-XOR
A multi-layer preceptron approach to solving the XOR Problem.

# XOR Problem
The XOR problem is the classic example of a linearly inseparable problem. It is also commonly used to show the limitation of a single-layer perceptron. In this repo, we would try to understand more deeply about the mathematics to why this is the case. 

# Multi-layer Perceptron
More specifically, we would want to understand how adding a hidden layer, with at least 3 hidden nodes, would allow the XOR problem to be solved. Hopefully, this could give us some insights into what is going under the hood in a mlp.

# Convexity Analysis
By playing the [tensorflow playground](http://playground.tensorflow.org), I have noticed that for a mlp with a hidden layer of three hidden nodes, the objective function is not convex with the backpropagation converge at different local minimums. However, when I add an additional hidden node, the objective function surprisingly become convex, with the backpropagation algorithm always converging. It would be interesting to understand why this is observed.
