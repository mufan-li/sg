'''
Script to test a subset of data
'''

import pandas as pd
import numpy as np
import numpy.linalg as la
import numpy.random as rd

import theano.tensor as T

# helper functions
from sg_functions import *
from mf import *
from rbm import *
from ae import *
from nnet2 import *

# # load and filter the data file
# execfile('preprocess.py')
sgdata_matrix = np.load('sgdata_matrix.npy')
sgDept_matrix = np.load('sgDept_matrix.npy')
missing_entries = np.load('missing_entries.npy')
sgMaj_matrix = np.load('sgMaj_matrix.npy')

# # matrix factorization
# execfile('mf_glrm.py')

# run_nnet(sgdata_matrix, sgMaj_matrix, learning_rate = 1e-3, training_epochs = 15,
# 		batch_size = 1000, v_hidden = [100, 100], momentum_const = 0.9, 
# 		cost_type = 'NLL')

# sgdata_predict_mf = run_mf(sgdata_matrix,
# 						learning_rate = 1e-5, 
# 						training_epochs = 20,
# 						d = 20, momentum_const = 0.2)
# summary(sgdata_predict_mf, sgdata_matrix, missing_entries, "MF")

# # RBM
# sgdata_predict_rbm, rbm = run_rbm(sgdata_matrix, 
# 						learning_rate = 1e-4, training_epochs = 20,
# 						n_hidden = 2, batch_size=50,
# 						rbm_class = RBM)
# summary(sgdata_predict_rbm, sgdata_matrix, missing_entries, "RBM")

# # AE
sg_predict_ae, sg_hid_ae = run_dA(sgdata_matrix,
					learning_rate = 1e-15, training_epochs = 1,
					n_hidden = 2, batch_size = 100,
					corruption_level = 0.3)
summary(sg_predict_ae, sgdata_matrix, missing_entries, "DAE")









