import numpy as np 
from regression import *
from scipy import linalg

class LinearRegression(Regression):
	"""
	Linear Regression Model:
	p = X @ W + e
	e ~ N(0,1)
	y = X @ W
	p ~	N(p|X @ W, Var)
	"""

	def fit(self, xArr : np.ndarray, yArr: np.ndarray):
		"""
		Linear Regression Algorithm: Least Squares Methods:
			W = [(X^T @ X)^(-1) @ X^T](Moore–Penrose inverse) @ p
		------
		Input:
		------
		 N: number of examples, D, dimension of the regressor
		1: xArr (N, D), np.ndarray, independent variable for training
		2. yArr (N, 1) np.ndarray, dependent variable for training
		xTx: (D, D)
		W:   (D, 1)
		------
		Output:
		------
		"""
		xMat = np.mat(xArr); yMat = np.mat(yArr).T
		xTx = xMat.T * xMat
		if linalg.det(xTx) == 0.0:
			print("This matirx is singular, connot do inverse")
		self.W = xTx.I * (xMat.T * yMat) 
		self.Var = np.mean(np.square(xMat * self.W - yMat))

	def predict(self, X : np.ndarray):
		"""
		Make Prediction Given X

		------
		Input:
		------
		N: number of examples, D, dimension of the regressor
		1. X (N, D) np.ndarray, independent variable for prediction
		
		------
		Output
		------
		1. Predicted y using Linear Regression
		2. Standard Deviation
		"""
		y = X * self.W
		std = np.sqrt(self.Var) + np.zeros_like(y)
		return y, std



lr = LinearRegression()
X = np.array([[1, 1, 1], [1, 2, 1], [2, 2, 1], [2, 3, 1]])
y = np.dot(X, np.array([1, 2, -1]))
lr.fit(X, y)



