# coding=utf-8

import matplotlib.pyplot as plt

from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset
from testCases_v2 import *


def layer_sizes(X, Y):
	"""

	:param X: input dataset of shape (input size, number of examples)
	:param Y: labels of shape (output size, number of examples)
	:return:
	n_x: the size of the input layer
	n_h: the size of the hidden layer
	n_y: the size of the output layer
	"""
	n_x = X.shape[0]  # size of input features
	n_h = 4  # size of hidden layers
	n_y = Y.shape[0]  # size of output layer
	return (n_x, n_h, n_y)


def initialize_parameters(n_x, n_h, n_y):
	"""
	:param n_x:
	:param n_h:
	:param n_y:
	
	:return:
	params -- python dictionary containing your parameters:
				W1 -- weight matrix of shape (n_h, n_y)
				b1 -- bias vector of shape (n_h, 1)
				W2 -- weight matrix of shape (n_y, n_h)
				b2 -- bias vector of shape (n_y, 1)
	"""
	W1 = np.random.randn(n_h, n_x) * 0.01
	# b1 = np.random.randn(n_h, 1)
	b1 = np.zeros((n_h, 1))
	W2 = np.random.randn(n_y, n_h)
	# b2 = np.random.randn(n_y, 1)
	b2 = np.zeros((n_y, 1))

	assert (W1.shape == (n_h, n_x))
	assert (b1.shape == (n_h, 1))
	assert (W2.shape == (n_y, n_h))
	assert (b2.shape == (n_y, 1))

	parameters = {"W1": W1,
	              "b1": b1,
	              "W2": W2,
	              "b2": b2}
	return parameters


def forward_propagation(X, parameters):
	"""
	:param X: input data of size (n_x, m)
	:param parameters: python dictionary containing your parameters (output of initialization function)

	:return: A2 -- The sigmoid output of the second activation. cache --a dictionary containing "Z1", "A1", "Z2", and "A2"
	"""
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]

	# TODO: forward propagation的实现还不知道对不对
	Z1 = W1.dot(X) + b1
	A1 = np.tanh(Z1)
	Z2 = W2.dot(A1) + b2
	A2 = sigmoid(Z2)

	assert (A2.shape == (1, X.shape[1]))

	cache = {"Z1": Z1,
	         "A1": A1,
	         "Z2": Z2,
	         "A2": A2}
	return A2, cache


def compute_cost(A2, Y, parameters):
	# TODO: parameters根本没有用到啊!
	m = Y.shape[1]
	cost = -1.0 / m * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
	cost = np.squeeze(cost)
	assert (isinstance(cost, float))
	return cost


def backward_propagation(parameters, cache, X, Y):
	"""
	Implement the backpropagation using the instructions above.

	:param parameters: python dictionary containing our parameters
	:param cache: a dictionary containing "Z1", "A1", "Z2", "A2"
	:param X: input data of shape (2, number of examples)
	:param Y: "true" labels vector of shape (1, number of examples)
	:return:
	grad -- python dictionary containing your gradients with respect to w and b

	"""
	m = X.shape[1]
	W1 = parameters["W1"]
	W2 = parameters["W2"]

	A1 = cache["A1"]
	A2 = cache["A2"]

	dZ2 = A2 - Y
	dW2 = 1.0 / m * dZ2.dot(A1.T)
	db2 = 1.0 / m * np.sum(dZ2, axis=1, keepdims=True)
	dZ1 = W2.T.dot(dZ2) * (1 - A1 ** 2)
	dW1 = 1.0 / m * dZ1.dot(X.T)
	db1 = 1.0 / m * np.sum(dZ1, axis=1, keepdims=True)

	grads = {"dW1": dW1,
	         "db1": db1,
	         "dW2": dW2,
	         "db2": db2}
	return grads


def update_parameters(parameters, grads, learning_rate=1.2):
	"""
	Update the parameters using the gradient descent rule given above

	:param parameters: python dictionary containing your parameters
	:param grads: python dictionary containing your gradients

	:return:
	:parameter: python dictionary containing your updated parameters
	"""
	W1 = parameters["W1"]
	W2 = parameters["W2"]
	b1 = parameters["b1"]
	b2 = parameters["b2"]

	dW1 = grads["dW1"]
	dW2 = grads["dW2"]
	db1 = grads["db1"]
	db2 = grads["db2"]

	# update rule for each parameters using gradient descent
	W1 = W1 - learning_rate * dW1
	W2 = W2 - learning_rate * dW2
	b1 = b1 - learning_rate * db1
	b2 = b2 - learning_rate * db2

	parameters = {"W1": W1,
	              "W2": W2,
	              "b1": b1,
	              "b2": b2}
	return parameters


def nn_model(X, Y, n_h, num_iterations=1000, print_cost=False):
	"""
	:param X: dataset of shape (2, number of examples)
	:param Y: dataset of shape (1, number of examples)
	:param n_h: size of the hidden layer
	:param num_iterations: number of iterations in gradient descent loop
	:param print_cost: if True, print the cost every 1000 iterations
	
	:return: 
	parameters: parameters learnt by the model. They can then be used to predict.
	"""
	np.random.seed(3)
	n_x = layer_sizes(X, Y)[0]
	n_y = layer_sizes(X, Y)[2]
	# Initialize parameters
	parameters = initialize_parameters(n_x, n_h, n_y)
	W1 = parameters["W1"]
	W2 = parameters["W2"]
	b1 = parameters["b1"]
	b2 = parameters["b2"]
	for i in range(0, num_iterations):
		# Forward propagation
		A2, cache = forward_propagation(X, parameters)
		# Cost function
		cost = compute_cost(A2, Y, parameters)
		# Backpropagation
		grads = backward_propagation(parameters, cache, X, Y)
		# Gradient descent parameter update
		parameters = update_parameters(parameters, grads)
		# print the cost every 1000 iterations
		if print_cost and i % 1000 == 0:
			print("Cost after iteration %i: %f" % (i, cost))
	return parameters


def predict(parameters, X):
	"""
	Using the learned parameters, predicts a class for each example in X

	:param parameters: python dictionary contraining your parameters
	:param X: input data of size (n_x, m)

	:return: predictions -- vector of predictions of our model (red: 0/blue: 1)
	"""
	A2, cache = forward_propagation(X, parameters)
	predictions = A2 > 0.5
	return predictions


def main():
	X, Y = load_planar_dataset()
	plt.figure(figsize=(16, 32))
	hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
	for i, n_h in enumerate(hidden_layer_sizes):
		plt.subplot(5, 2, i + 1)
		plt.title("Hidden layer of size: %d" % n_h)
		parameters = nn_model(X, Y, n_h, num_iterations=5000)
		plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
		predictions = predict(parameters, X)
		accuracy = np.sum(predictions == Y) / float(Y.shape[1]) * 100.0
		print("Accuracy for {} hidden units: {}%".format(n_h, accuracy))
	plt.show()


if __name__ == "__main__":
	main()
