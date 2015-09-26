'''
Implement related helper functions
'''

import numpy as np
import numpy.linalg as la

def find_missing_entries(A, missing_val = 0):
	# input: 
	#	A - a matrix A with nan as its missing entires
	# outputs:
	# 	A - a matrix A with its missing entries replaced
	#	A_miss - a list of tuples for all missing indicies

	v_miss = np.where(np.isnan(A))
	A_miss = [(v1,v2) for (v1,v2) in zip(v_miss[0],v_miss[1])]
	A[v_miss] = missing_val

	return A, A_miss

def fbnorm(A, A_miss):
	# inputs:
	# 	A - a matrix A with missing entries
	#	A_miss - missing entries of A
	# outputs:
	#	fn - the Frobenius norm with missing entries

	v_miss = (np.asarray([v[0] for v in A_miss]), \
				np.asarray([v[1] for v in A_miss]))
	A[v_miss] = 0
	fn = la.norm(A)

	return fn

