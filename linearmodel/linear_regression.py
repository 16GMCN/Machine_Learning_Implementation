import numpy as np 
from linearmodel.regression import Regression 

class LinearRegression(Regression):
	"""
	Linear Regression Model:
	p = X @ W + e
	e ~ N(0,1)
	y = X @ W
	p ~	N(p|X @ W, Var)
	"""

	def fit(self, X : np.ndarray, p: np.ndarray):
		"""
		Linear Regression Algorithm: Least Squares Methods:
			W = [(X^T @ X)^(-1) @ X^T](Moore–Penrose inverse) @ p
		------
		Input:
		------
		 N: number of examples, D, dimension of the regressor
		1: X (N, D), np.ndarray, independent variable for training
		2. p (N,) np.ndarray, dependent variable for training
		------
		Output:
		------
		"""
		self.W = np.linalg.pinv(X) @ p 
		self.Var = np.mean(np.square(X @ self.W - p))

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
		y = X @ self.W
		std = np.sqrt(self.Var) + np.zeros_like(y)
		return y, std