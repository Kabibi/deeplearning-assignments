# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np

from lr_utils import load_dataset


def preprocessing():
	"""
	Load the data from file and preprocess

	:return:
	train_set_x: preprocessed training data
	train_set_y: preprocessed testing label
	test_set_x: preprocessed testing data
	test_set_y: preprocessed testing label
	"""
	train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

	m_train = train_set_x_orig.shape[0]
	m_test = test_set_x_orig.shape[0]
	num_px = train_set_x_orig.shape[1]

	train_set_x_flatten = train_set_x_orig.reshape((m_train, num_px * num_px * 3)).T
	test_set_x_flatten = test_set_x_orig.reshape((m_test, num_px * num_px * 3)).T

	train_set_x = train_set_x_flatten / 255.
	test_set_x = test_set_x_flatten / 255.

	return train_set_x, train_set_y, test_set_x, test_set_y


def sigmoid(x):
	return 1.0 / (1 + np.exp(-x))


def initialize_with_zeros(dim):
	"""
	This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

	Argument:
	dim -- size of the w vector we want (or number of parameters in this case)

	Returns:
	w -- initialized vector of shape (dim, 1)
	b -- initialized scalar (corresponds to the bias)
	"""
	w = np.zeros((dim, 1))
	b = 0
	assert (w.shape == (dim, 1))
	assert (isinstance(b, float) or isinstance(b, int))
	return w, b


def propagate(w, b, X, Y):
	"""
	Implement the cost function and its gradient for the propagation explained above

	:param w: weights, a numpy array of size (num_px*num_px*3, 1)
	:param b: bias, a scalar
	:param X: data of size (num_px*num_px*3, number of examples)
	:param Y: true "label" vector (containing 0 if non-cat, 1 if cat) of size(1, number of examples)
	:return:
	:cost -- negative log-likelihood cost for logistic regression
	:dw -- graident of the loss with respect to w, thus same shape as w
	:db -- graident of the loss with respect to b, thus same shape as b
	"""
	m = X.shape[1]
	A = sigmoid(w.T.dot(X) + b)
	cost = -1.0 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
	dw = 1.0 / m * X.dot((A - Y).T)
	db = 1.0 / m * np.sum(A - Y)
	assert (dw.shape == w.shape)
	assert (db.dtype == float)
	cost = np.squeeze(cost)
	assert (cost.shape == ())
	grads = {"dw": dw, "db": db}
	return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
	"""
	This function optimizes w and b by running a gradient descent algorithm

	:param w: weights, a numpy array of size (num_px*num_px*3, 1)
	:param b: bias, a scalar
	:param X: data of shape (num_px*num_px*3, number of examples)
	:param Y: true "label" vector (containing 0 if non-cat, 1 if cat), of shape(1, number of examples)
	:param num_iterations: number of iterations of the optimization loop
	:param learning_rate: learning rate of the gradient descent updata rule
	:param print_cost: True to print the loss every 100 steps
	:return:
	params: dictionary containing the weights w and bias b
	grads: dictionary containing the gradients of the weights and bias with respect to the cost function
	costs: list of all the costs computed during the optimization, this will be used to plot the learning curve
	"""
	costs = []
	for i in range(num_iterations):
		grads, cost = propagate(w, b, X, Y)
		dw = grads["dw"]
		db = grads["db"]
		w = w - learning_rate * dw
		b = b - learning_rate * db
		if i % 100 == 0:
			costs.append(cost)
		if print_cost and i % 100 == 0:
			print("Cost after iteration %i: %f" % (i, cost))
	params = {"w": w, "b": b}
	grads = {"dw": dw, "db": db}
	return params, grads, costs


def predict(w, b, X):
	"""
	Predict whether the label is 0 or 1 using learned logistic regression parameters(w, b)

	:param w: weights, a numpy array of size (num_px*num_px*3, 1)
	:param b: bias, a scalar
	:param X: data of size (num_px*num_px*3, number of examples)
	:return: Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
	"""
	m = X.shape[1]
	w = w.reshape(X.shape[0], 1)

	# Compute vector "A" predicting the probabilities of a cat being present in the picture
	A = sigmoid(w.T.dot(X) + b)
	Y_prediction = (A > 0.5) + 0
	assert (Y_prediction.shape == (1, m))
	return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
	"""
	Build the logistic regression model by calling the function you've implemented previously

	:param X_trian: training set represented by numpy array of shape (num_px*num_px*3, m_train)
	:param Y_train: trainging labels represented by a numpy array (vector) of shape (1, m_train)
	:param X_test: test set represented by a numpy array (vector) of shape (1, m_test)
	:param Y_test: test labels represented by a numpy array (vector) of shape (1, m_test)
	:param num_iterations: hyperparameter representing the learning rate to optimize the parameters
	:param learning_rate: hyperparameter representing the learning rate used in the update rule of optimize()
	:param print_cost: Set to true to print the cost every 100 iterations
	:return:
	d -- dictionary containing information about the model
	"""
	w, b = initialize_with_zeros(X_train.shape[0])

	parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost=True)

	w = parameters["w"]
	b = parameters["b"]

	Y_prediction_train = predict(w, b, X_train)
	Y_prediction_test = predict(w, b, X_test)

	print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
	print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

	d = {"costs":              costs,
	     "Y_prediction_test":  Y_prediction_test,
	     "Y_prediction_train": Y_prediction_train,
	     "w":                  w,
	     "b":                  b,
	     "learning_rate":      learning_rate,
	     "num_iterations":     num_iterations}
	return d


def main():
	train_set_x, train_set_y, test_set_x, test_set_y = preprocessing()

	d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=200, learning_rate=0.005,
	          print_cost=True)

	# try different learning rate
	learning_rate = [0.01, 0.001, 0.0001]
	models = {}
	for i in learning_rate:
		print("learning rate is: " + str(i))
		models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=i,
		                       print_cost=False)
	print("\n" + "--------------------------------------" + '\n')

	# plot cost function
	for i in learning_rate:
		plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

	plt.ylabel('cost')
	plt.xlabel('iterations')

	legend = plt.legend(loc='upper center', shadow=True)
	frame = legend.get_frame()
	frame.set_facecolor('0.90')
	plt.show()


if __name__ == "__main__":
	main()
