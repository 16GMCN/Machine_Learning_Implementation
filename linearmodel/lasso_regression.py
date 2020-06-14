import numpy as np 
from regression import *
from scipy import linalg
import itertools


class LassoRegression(Regression):
	"""
	RSS = np.sum(np.square(Y - X*W))
	coordinate descent
	"""
	def fit(self, X:np.array, Y:np.array, W:np.array, lambd = 0.1):
		rss = lambda X, y, w: (y - X * w).T * (y - X * w)
		N, D = X.shape 
		W = np.matrix(np.zeros((D, 1)))
		r = rss(X, Y, W)
		niter = itertools.count(1)
		for it in niter:
			for k in range(D):
				z_k = (X[:, k].T * X[:, k])[0, 0]
				p_k = 0
				for i in range(N):
					p_k += X[i, k]*(y[i, 0] - sum([X[i, j]*w[j, 0] for j in range(n) if j != k]))
				if p_k < -lambd / 2:
					w_k = (p_k + lambd / 2) / z_k
				elif p_k > lambd / 2;
					w_k = (p_k - lambd / 2) / z_k
				else:
					w_k = 0
				w[k, 0] = w_k
			r_prime = rss(X, Y, W)
			delta = abs(r_prime - r)[0, 0]
			r = r_prime
			if delta < threshold:
				break
		self.W = W
		self.Var = np.mean(np.square(Y - X * self.W))





