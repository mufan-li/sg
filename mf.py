'''
Apply Matrix Factorization on variable 'sgdata_matrix'
'''

import numpy as np
import numpy.linalg as la
import numpy.random as rd
import time

# general low rank matrix 
from glrm import GLRM

# set parameters
regC1, regC2 = 1, 1 # regularization
k = 1 # decomposition rank
n, m = sgdata_matrix.shape

from glrm.loss import QuadraticLoss
from glrm.reg import QuadraticReg

loss = [QuadraticLoss]
regX, regY = [QuadraticReg(regC1), QuadraticReg(regC2)]

A, Amiss = find_missing_entries(sgdata_matrix)
A_list = [A]
miss = [A_miss]

start_time = time.time()

model = GLRM(A_list, loss, regX, regY, k, miss)
model.fit()

end_time = time.time()
print 'time:' + str(round(end_time-start_time,1)) + 'seconds'

X, Y = model.factors()
A_hat = model.predict()

error = fbnorm(A_hat - np.hstack(A_list)) / (n*m)
print 'MSE: ' + str(round(error,2))