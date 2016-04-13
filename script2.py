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
sgdata_matrix_ly = np.load('sgdata_matrix_ly.npy')
sgdata_matrix_uy = np.load('sgdata_matrix_uy.npy')
sgDept_matrix = np.load('sgDept_matrix.npy')
missing_entries = np.load('missing_entries.npy')
sgMaj_matrix = np.load('sgMaj_matrix.npy')

# # matrix factorization
# execfile('mf_glrm.py')

# binary

sgdata_matrix_ly[sgdata_matrix_ly==0] = -1
# sgdata_matrix_ly = np.exp((sgdata_matrix_ly - 0.5)*5)
# sgdata_matrix_ly = (sgdata_matrix_ly>0).astype(int)
sgdata_matrix_uy = (sgdata_matrix_uy>0).astype(int)

# Predict Majors
sgMaj_pred, sgMaj_train_MSE, sgMaj_test_MSE, sgMaj_train_error_rate, \
	sgMaj_test_error_rate = run_nnet(
		sgdata_matrix_ly, sgMaj_matrix, 
		learning_rate = 1e-3, training_epochs = 50,
		batch_size = 100, 
		v_hidden = [500,500],
		momentum_const = 0, 
		cost_type = 'NLL', 
		actv_fcn = relu,
		# out_actv_fcn = T.nnet.sigmoid,
		dropout_rate = 0.3, lr_decay = 0,
		update_method = 'adam')

nn_plot_results(sgMaj_train_MSE, sgMaj_test_MSE, 
	sgMaj_train_error_rate, sgMaj_test_error_rate)

# # Predict Course Selection
# sguy_pred, sguy_train_MSE, sguy_test_MSE, sguy_train_error_rate, \
# 	sguy_test_error_rate = run_nnet(
# 		sgdata_matrix_ly, sgdata_matrix_uy, 
# 		learning_rate = 1e-1, training_epochs = 100,
# 		batch_size = 50, 
# 		v_hidden = [200,200,200,200,200,200],
# 		momentum_const = 0, 
# 		cost_type = 'NLL', 
# 		actv_fcn = relu,
# 		out_actv_fcn = T.nnet.sigmoid,
# 		dropout_rate = 0.3, lr_decay = 0.01,
# 		pred_course = True,
# 		update_method = 'adam')

# print 'Courses Taken: ', \
# 	np.mean(1-sguy_pred[sgdata_matrix_uy.astype(bool)])
# print 'Courses Not Taken: ', \
# 	np.mean(1-sguy_pred[(1-sgdata_matrix_uy).astype(bool)])

# print np.round(sgMaj_pred[:10])
# print sgMaj_matrix[:10]

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
# sg_predict_ae, sg_hid_ae = run_dA(sgdata_matrix,
# 					learning_rate = 1e-2, training_epochs = 100,
# 					n_hidden = 100, batch_size = 20,
# 					corruption_level = 0.3,
# 					actv_fcn = relu)
# summary(sg_predict_ae, sgdata_matrix, missing_entries, "DAE")

# import matplotlib.pyplot as plt
# sg_hid_ae = np.load('sg_hid_ae.npy')
# plt.scatter(sg_hid_ae[:,0], sg_hid_ae[:,1], \
# 	s=30, c=np.sum(sgdata_matrix,1) / np.sum(~missing_entries,1), 
			# alpha=0.5)
# plt.show()

# plt.scatter(sg_hid_ae[:,0], sg_hid_ae[:,1], \
# 	s=30, c=sgMaj_matrix[:,43], alpha=0.3)
# plt.show()







