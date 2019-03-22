import pandas as pd
import numpy as np
import math

from metrics import accuracy_score, crossEntropy_error, RMS_error, confusion_matrix


class LogisticRegression:

	def __init__(self):
		self.training_error = None
		self.training_accuracy = None
		self.M = None
		self.K = None
		self.W = None
		self.b = None
		
	def fit(self, X0, T0, Eta, epochs, batchSize):
		X = X0.copy()
		T = T0.copy()
		self.M = len(X)
		self.K = len(T)
		N = len(X[0])
		self.training_error = []
		self.training_accuracy = []
		b = np.zeros((self.K, batchSize))
		self.W = np.zeros((self.M, self.K))
		for e in range(0, epochs):
			for i in range(0, N, batchSize):
				x = X[:,i:(i+batchSize)]
				t = T[:,i:(i+batchSize)]
				a = np.dot(np.transpose(self.W), x) + b
				a_exp = np.exp(a)
				a_exp_Sum = (a_exp).sum(axis=0)
				a_exp_Sum = (np.transpose(a_exp_Sum)).reshape((1, batchSize))
				y = np.transpose(np.transpose(a_exp) / np.transpose(a_exp_Sum))
				diff = (np.transpose(y - t)).reshape((batchSize, self.K))
				deltaE = np.dot( x, diff )
				deltaW = -1 * Eta * deltaE
				self.W = self.W + deltaW
			Y = np.transpose(self.predict(X))
			err = crossEntropy_error( np.transpose(T), Y )
			(self.training_error).append(err)
			YF = np.transpose(self.classificationPredict(np.transpose(Y)))
			acc = accuracy_score( np.transpose(T), YF )
			(self.training_accuracy).append(acc)
		self.training_error = np.array(self.training_error)
		self.training_accuracy = np.array(self.training_accuracy)
		self.b = b[:,0]

	def predict(self, X0):
		X = X0.copy()
		A = np.dot((np.transpose(self.W)), X)
		A_Exp = np.exp(A)
		A_Exp_Sum = (np.transpose(A_Exp)).sum(axis=1)
		Y = A_Exp / A_Exp_Sum
		return Y

	def classificationPredict(self, Y0):
		Y = Y0.copy()
		Y1 = np.zeros((len(Y), len(Y[0])))
		i=0
		while(i<len(Y[0])):
			Y1[(np.argmax(Y[:,i]))][i] = 1.0
			i=i+1
		return Y1