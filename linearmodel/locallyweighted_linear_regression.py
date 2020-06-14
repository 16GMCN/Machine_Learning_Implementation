import numpy as np 
from regression import *
from scipy import linalg

class LwlRegression(Regression):
	"""
	testPoint: (1, D)
	xMat: (N, D)
	yMat: (N, 1)
	W: (D,1)
	"""
	def fit(self, testPoint: np.array, xArray : np.array, yArray : np.array, k = 1.0 ):
		xMat = np.mat(xArray); yMat = np.mat(yArray).T
		N = shape(xMat)[0]
		weights = np.mat(np.eye(N))
		for j in range(N):
			diffMat = testPoint - xMat[j, :]
			weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
		xTx = xMat.T * (weights * xMat)
		if linalg.det(xTx) == 0.0:
			print("This matrix is singular, cannot do inverse")

		self.W = xTx.I * (xMat.T * (weights * yMat))
		self.Var = np.mean(np.square(yMat - xMat * self.W))

	def predict(self, X: np.array):
		
		y = X * self.W
		std = np.sqrt(self.Var) + np.zeros_like(y)
		return y, std
