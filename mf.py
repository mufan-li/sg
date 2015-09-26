'''
Apply Matrix Factorization on variable 'sgdata_matrix'
'''

import numpy as np
import numpy.linalg as la
import numpy.random as rd
import pylab as P
import time

# general low rank matrix 
from glrm import GLRM
# helper functions
from sg_functions import *

# set parameters
regC1, regC2 = 1, 1 # regularization
k = 20 # decomposition rank
N, M = sgdata_matrix.shape

from glrm.loss import QuadraticLoss
from glrm.reg import QuadraticReg

loss = [QuadraticLoss]
regX, regY = [QuadraticReg(regC1), QuadraticReg(regC2)]

A, A_miss = find_missing_entries(sgdata_matrix)
A_list = [A]
miss = [A_miss]

start_time = time.time()

model = GLRM(A_list, loss, regX, regY, k, miss)
model.fit()

end_time = time.time()
print 'time:' + str(round(end_time-start_time,1)) + 'seconds'

X, Y = model.factors()
A_hat = model.predict()

error = fbnorm(A_hat - np.hstack(A_list), A_miss)
print 'Frobenius Error: ' + str(round(error,2))

ind = np.where(A>0)
hData = abs(A-A_hat).round(0)[ind]

n, bins, patches = P.hist(hData, 20, normed=1, histtype='stepfilled')
P.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
# P.figure()
P.show()







