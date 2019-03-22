import numpy as np
import math


def RMS_error(T, Y):
	diffSq = np.square(T - Y)
	diffSq_sum = np.sum(diffSq)
	n = len(Y)
	return math.sqrt(diffSq_sum/n)


def crossEntropy_error(T, Y):
	sum = 0
	i=0
	Y_ln = np.log(Y)
	N = len(Y_ln)
	K = len(Y_ln[0])
	while(i<N):
		e = 0
		j=0
		while(j<K):
			e = e + T[i][j]*Y_ln[i][j]
			j=j+1
		e = -1 * e
		sum = sum + e
		i=i+1
	avg = (sum/N)
	return avg


def accuracy_score(T, Y):
	cnt = 0
	i=0
	while(i<len(T)):
		if ( (T.ndim==1 and T[i]==Y[i]) or (T.ndim==2 and np.array_equal(T[i],Y[i])) ):
			cnt=cnt+1
		i=i+1
	return ((cnt/len(T))*100)


def confusion_matrix(T, Y):
	CM = np.zeros((10, 10))
	K = 10
	i=0
	while(i<len(T)):
		if(T.ndim==1):
			t = T[i]
			y = Y[i]
			CM[t][y] = CM[t][y] + 1
		else:
			t = np.argmax(T[i])
			y = np.argmax(Y[i])
			CM[t][y] = CM[t][y] + 1
		i=i+1
	k = 0
	s = 0
	while ( k < K ):
		s = s + CM[k][k]
		k = k + 1
	AS = (s/len(T))*100
	return CM, AS