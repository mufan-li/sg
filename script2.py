'''
Script to test a subset of data
'''

import pandas as pd
import numpy as np
import numpy.linalg as la
import numpy.random as rd

# helper functions
from sg_functions import *
from rbm import *

# load and filter the data file
print '... prepocessing data'
execfile('preprocess.py')

# filter for repeated courses
# use group.last() assuming the last is most recent
sgGroup = sgdata[['ID','COURSE','GRADE']].groupby(['ID', 'COURSE'])

sgdataFilter = sgGroup.last().reset_index()
sgdata_pivot = sgdataFilter.pivot(index='ID', \
				columns='COURSE', values='GRADE')

sgdata_matrix = np.asarray(sgdata_pivot)
# set to zero
missing_entries = np.isnan(sgdata_matrix)
sgdata_matrix[missing_entries] = 0

# matrix factorization
# execfile('mf.py')

sgdata_matrix = sgdata_matrix/100. # rescale to [0,1]
# RBM
sgdata_predict, RBM = run_rbm(sgdata_matrix, 
						learning_rate = 1e-5, training_epochs = 100,
						n_hidden = 100,
						batch_size=50)
# print sgdata_matrix[:5,:5]
# print sgdata_predict[:5,:5]
print sgdata_matrix[~missing_entries][:10]
print sgdata_predict[~missing_entries][:10].round(2)

idx = sgdata_matrix.shape[0]/5
rbm_error = np.sqrt(np.mean(np.square(
	(sgdata_predict - sgdata_matrix)[:4*idx,:][~missing_entries[:4*idx,:]]
	)))

print 'RBM RMSE: ', rbm_error










