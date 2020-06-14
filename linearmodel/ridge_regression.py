import numpy as np 
from regression import *
from scipy import linalg

class RidgeRegression(Regression):
	"""
	xMat: (N, D)
	yMat: (N, 1)
	W: (D,1)
	Lambda: scalar
	"""
	def fit(self,  xArray : np.array, yArray : np.array, lam = 0.2 ):
		xMat = np.mat(xArray); yMat = np.mat(yArray).T
		xTx = xMat.T * xMat
		denom = xTx + np.eye(shape(xMat)[1]) * lam 

		if linalg.det(denom) == 0.0:
			print("This matrix is singular, cannot do inverse")

		self.W = denom.I * (xMat.T  * yMat)
		self.Var = np.mean(np.square(yMat - xMat * self.W))

	def predict(self, X: np.array):
		
		y = X * self.W
		std = np.sqrt(self.Var) + np.zeros_like(y)
		return y, std
