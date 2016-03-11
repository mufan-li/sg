'''
Implement related helper functions
'''

import numpy as np
import numpy.linalg as la
import os
import gzip
import cPickle
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

def find_missing_entries(A, missing_val = 0):
	# input: 
	#	A - a matrix A with nan as its missing entires
	# outputs:
	# 	A - a matrix A with its missing entries replaced
	#	A_miss - a list of tuples for all missing indicies

	v_miss = (A==0)
	v2_miss = np.where(v_miss)
	A_miss = [(v1,v2) for (v1,v2) in zip(v2_miss[0],v2_miss[1])]
	A[v_miss] = missing_val

	return A, A_miss, v_miss

def fbnorm(A, v_miss):
	# inputs:
	# 	A - a matrix A with missing entries
	#	v_miss - missing entries of A
	# outputs:
	#	fn - the Frobenius norm with missing entries

	A[v_miss] = 0
	fn = la.norm(A)

	return fn

def rmse(A, A_hat, v_miss):
	# inputs:
	# 	A - a matrix A with missing entries
	#	v_miss - missing entries of A
	# outputs:
	#	er - the RMSE of A with missing entries
	return np.sqrt(np.mean(np.square((A - A_hat)[~v_miss])))

def shared_data(data_x, borrow=True):
		''' share only one data variable
		'''
		# data_x
		shared_x = theano.shared(numpy.asarray(data_x,
									   dtype=theano.config.floatX),
								 borrow=borrow)
		return shared_x


def train_f_select(epoch, training_epochs, f_list):
	''' selects CD-k based on epoch
	'''
	# inputs:
	# 	epoch - current epoch
	#	training_epochs - total number of epochs
	# 	f_list - list of functions to choose from

	n_functions = len(f_list)
	n_per_function = np.ceil(training_epochs*1.0 / n_functions)
	ind = epoch/n_per_function

	return f_list[ind.astype(int)]

def load_mnist_data(dataset):
	''' Loads the dataset

	:type dataset: string
	:param dataset: the path to the dataset (here MNIST)
	'''

	#############
	# LOAD DATA #
	#############

	# Download the MNIST dataset if it is not present
	data_dir, data_file = os.path.split(dataset)
	if data_dir == "" and not os.path.isfile(dataset):
		# Check if dataset is in the data directory.
		new_path = os.path.join(
			os.path.split(__file__)[0],
			"..",
			"data",
			dataset
		)
		if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
			dataset = new_path

	if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
		import urllib
		origin = (
		'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
		)
		print 'Downloading data from %s' % origin
		urllib.urlretrieve(origin, dataset)

	print '... loading data'

	# Load the dataset
	f = gzip.open(dataset, 'rb')
	train_set, valid_set, test_set = cPickle.load(f)
	f.close()
	#train_set, valid_set, test_set format: tuple(input, target)
	#input is an numpy.ndarray of 2 dimensions (a matrix)
	#witch row's correspond to an example. target is a
	#numpy.ndarray of 1 dimensions (vector)) that have the same length as
	#the number of rows in the input. It should give the target
	#target to the example with the same index in the input.

	def shared_dataset(data_xy, borrow=True):
		""" Function that loads the dataset into shared variables

		The reason we store our dataset in shared variables is to allow
		Theano to copy it into the GPU memory (when code is run on GPU).
		Since copying data into the GPU is slow, copying a minibatch 
		everytime is needed (the default behaviour if the data is not 
		in a shared variable) would lead to a large decrease in 
		performance.
		"""
		data_x, data_y = data_xy
		shared_x = theano.shared(numpy.asarray(data_x,
										   dtype=theano.config.floatX),
								 borrow=borrow)
		shared_y = theano.shared(numpy.asarray(data_y,
										   dtype=theano.config.floatX),
								 borrow=borrow)
		# When storing data on the GPU it has to be stored as floats
		# therefore we will store the labels as ``floatX`` as well
		# (``shared_y`` does exactly that). But during our computations
		# we need them as ints (we use labels as index, and if they are
		# floats it doesn't make sense) therefore instead of returning
		# ``shared_y`` we will have to cast it to int. This little hack
		# lets ous get around this issue
		return shared_x, T.cast(shared_y, 'int32')

	test_set_x, test_set_y = shared_dataset(test_set)
	valid_set_x, valid_set_y = shared_dataset(valid_set)
	train_set_x, train_set_y = shared_dataset(train_set)

	rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
			(test_set_x, test_set_y)]
	return rval


def identity_fn(x):
	''' returns the identity
	'''
	return x


def get_major_prop(sample_df, min_credit = 3):
	''' returns the proportion department of major
	'''
	total = sum(sample_df.CREDIT_TOTAL)
	sample_df['DEPT_PROP'] = sample_df['CREDIT_TOTAL'] / total
	return sample_df
	
def summary(sgdata_predict, sgdata_matrix, 
		missing_entries, name):
	print sgdata_matrix[~missing_entries][:10]
	print sgdata_predict[~missing_entries][:10].round(2)

	idx = sgdata_matrix.shape[0]/5
	error = np.sqrt(np.mean(np.square(
		(sgdata_predict - sgdata_matrix)[4*idx:,:][\
			~missing_entries[4*idx:,:]]
		)))

	print name, ' RMSE: ', error

def relu(x):
    return theano.tensor.switch(x<0, 0, x)


def nn_plot_results(sgMaj_train_MSE, sgMaj_test_MSE, 
	sgMaj_train_error_rate, sgMaj_test_error_rate):
	plt.figure(1)

	plt.subplot(211)
	x_axis = np.arange(len(sgMaj_train_MSE))
	plt.plot(x_axis,sgMaj_train_MSE,label="Training Data")
	plt.plot(x_axis,sgMaj_test_MSE,label="Test Data")
	# plt.xlabel('Training Epoch')
	plt.ylabel('Negative Log-Likelihood')
	plt.title('Training Progress')
	plt.grid(True)
	# plt.yscale('log')
	plt.legend()

	plt.subplot(212)
	plt.plot(x_axis, sgMaj_train_error_rate, label='Training Data')
	plt.plot(x_axis, sgMaj_test_error_rate,label='Test Data')
	plt.xlabel('Training Epoch')
	plt.ylabel('Error Rate')
	# plt.title('Training Progress - Error')
	plt.grid(True)
	# plt.yscale('log')
	plt.legend()
	plt.show()






















""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""


import numpy


def scale_to_unit_interval(ndar, eps=1e-8):
	""" Scales all values in the ndarray ndar to be between 0 and 1 """
	ndar = ndar.copy()
	ndar -= ndar.min()
	ndar *= 1.0 / (ndar.max() + eps)
	return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
					   scale_rows_to_unit_interval=True,
					   output_pixel_vals=True):
	"""
	Transform an array with one flattened image per row, into an array in
	which images are reshaped and layed out like tiles on a floor.

	This function is useful for visualizing datasets whose rows are images,
	and also columns of matrices for transforming those rows
	(such as the first layer of a neural net).

	:type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
	be 2-D ndarrays or None;
	:param X: a 2-D array in which every row is a flattened image.

	:type img_shape: tuple; (height, width)
	:param img_shape: the original shape of each image

	:type tile_shape: tuple; (rows, cols)
	:param tile_shape: the number of images to tile (rows, cols)

	:param output_pixel_vals: if output should be pixel values (i.e. int8
	values) or floats

	:param scale_rows_to_unit_interval: if the values need to be scaled before
	being plotted to [0,1] or not


	:returns: array suitable for viewing as an image.
	(See:`Image.fromarray`.)
	:rtype: a 2-d array with same dtype as X.

	"""

	assert len(img_shape) == 2
	assert len(tile_shape) == 2
	assert len(tile_spacing) == 2

	# The expression below can be re-written in a more C style as
	# follows :
	#
	# out_shape	= [0,0]
	# out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
	#				tile_spacing[0]
	# out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
	#				tile_spacing[1]
	out_shape = [
		(ishp + tsp) * tshp - tsp
		for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
	]

	if isinstance(X, tuple):
		assert len(X) == 4
		# Create an output numpy ndarray to store the image
		if output_pixel_vals:
			out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
									dtype='uint8')
		else:
			out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
									dtype=X.dtype)

		#colors default to 0, alpha defaults to 1 (opaque)
		if output_pixel_vals:
			channel_defaults = [0, 0, 0, 255]
		else:
			channel_defaults = [0., 0., 0., 1.]

		for i in xrange(4):
			if X[i] is None:
				# if channel is None, fill it with zeros of the correct
				# dtype
				dt = out_array.dtype
				if output_pixel_vals:
					dt = 'uint8'
				out_array[:, :, i] = numpy.zeros(
					out_shape,
					dtype=dt
				) + channel_defaults[i]
			else:
				# use a recurrent call to compute the channel and store it
				# in the output
				out_array[:, :, i] = tile_raster_images(
					X[i], img_shape, tile_shape, tile_spacing,
					scale_rows_to_unit_interval, output_pixel_vals)
		return out_array

	else:
		# if we are dealing with only one channel
		H, W = img_shape
		Hs, Ws = tile_spacing

		# generate a matrix to store the output
		dt = X.dtype
		if output_pixel_vals:
			dt = 'uint8'
		out_array = numpy.zeros(out_shape, dtype=dt)

		for tile_row in xrange(tile_shape[0]):
			for tile_col in xrange(tile_shape[1]):
				if tile_row * tile_shape[1] + tile_col < X.shape[0]:
					this_x = X[tile_row * tile_shape[1] + tile_col]
					if scale_rows_to_unit_interval:
						# if we should scale values to be between 0 and 1
						# do this by calling the `scale_to_unit_interval`
						# function
						this_img = scale_to_unit_interval(
							this_x.reshape(img_shape))
					else:
						this_img = this_x.reshape(img_shape)
					# add the slice to the corresponding position in the
					# output array
					c = 1
					if output_pixel_vals:
						c = 255
					out_array[
						tile_row * (H + Hs): tile_row * (H + Hs) + H,
						tile_col * (W + Ws): tile_col * (W + Ws) + W
					] = this_img * c
		return out_array