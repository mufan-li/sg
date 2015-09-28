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
execfile('preprocess.py')

# filter for repeated courses
# use group.last() assuming the last is most recent
sgGroup = sgdata[['ID','COURSE','GRADE']].groupby(['ID', 'COURSE'])

sgdataFilter = sgGroup.last().reset_index()
sgdata_pivot = sgdataFilter.pivot(index='ID', \
				columns='COURSE', values='GRADE')

sgdata_matrix = np.asarray(sgdata_pivot)
# set to zero
sgdata_matrix[np.isnan(sgdata_matrix)] = 0
sgdata_matrix = sgdata_matrix/100. # rescale to [0,1]

# matrix factorization
# execfile('mf.py')

# RBM
run_rbm(sgdata_matrix, training_epochs = 100)