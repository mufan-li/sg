
import timeit

import numpy as np
import numpy.random as rd

import theano
import theano.tensor as T
import os

from theano.tensor.shared_randomstreams import RandomStreams

from sg_functions import *

'''
############
# MF class #
############
'''

class MF(object):
	def __init__(
		self,
		input=None,
		input_nm=None,
		n_class=784,
		n_student=500,
		d=5,
		A=None,
		B=None,
		numpy_rng=None,
		theano_rng=None
	):
		self.n_class = n_class
		self.n_student = n_student

		if numpy_rng is None:
			# create a numpy generator
			numpy_rng = rd.RandomState(1234)

		if theano_rng is None:
			theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

		# need to create shared variables
		if A is None:
			initial_A = np.asarray(
				numpy_rng.uniform(
					low=-4 * np.sqrt(6. / (n_student)),
					high=4 * np.sqrt(6. / (n_student)),
					size=(n_student, d)
				),
				dtype = theano.config.floatX
			)
			A = theano.shared(value=initial_A, name='A', borrow=True)

		if B is None:
			initial_B = np.asarray(
				numpy_rng.uniform(
					low=-4 * np.sqrt(6. / (n_class)),
					high=4 * np.sqrt(6. / (n_class)),
					size=(d, n_class)
				),
				dtype = theano.config.floatX
			)
			B = theano.shared(value=initial_B, name='B', borrow=True)

		self.input = input
		if not input:
			self.input = T.matrix('input')
		self.input_nm = input_nm
		if not input_nm:
			self.input = T.matrix('input_nm')

		self.A = A
		self.B = B
		self.theano_rng = theano_rng
		self.params = [self.A, self.B]

		self.predict = T.dot(A,B)

def run_mf(dataset, learning_rate = 0.1, training_epochs = 10,
		d = 5, momentum_const = 0.9):
	theano.config.on_unused_input = 'warn'

	train_idx = dataset.shape[0]*4/5
	train_set_x = shared_data(dataset[:train_idx, :])
	train_not_miss = shared_data( ~(dataset[:train_idx, :]==0) )
	test_set_x = shared_data(dataset[train_idx:, :])
	test_not_miss = shared_data( ~(dataset[train_idx:, :]==0) )

	missing_entries = shared_data(~(dataset==0))
	complete_set = shared_data(dataset)

	x = T.matrix('x')
	xnm = T.matrix('xnm') # for not_miss matrix

	rng = numpy.random.RandomState(123)
	theano_rng = RandomStreams(rng.randint(2 ** 30))

	mf = MF(
		input=x,
		input_nm=xnm,
		n_class=dataset.shape[1],
		n_student=dataset.shape[0],
		d=5
	)

	cost = T.sum(T.square( mf.input - mf.predict * xnm )) \
		/ T.sum(mf.input_nm)

	gparams = [T.grad(cost, param) for param in mf.params]

	vparams = [theano.shared(param.get_value(), borrow=True)
				for param in mf.params
	]
	update1 = [
		(vparam, momentum_const * vparam - learning_rate * gparam) 
		for vparam, gparam in zip(vparams, gparams)
	]
	update2 = [
		(param, param + vparam) 
		for param, vparam in zip(mf.params, vparams)
	]

	print '... building training function ...'
	train_mf = theano.function(
		[],
		cost,
		updates = update1 + update2,
		givens = {
			x: complete_set,
			xnm: missing_entries
		},
		name = 'train_mf'
	)

	print '... building predict function ...'
	predict = theano.function(
		[x],
		mf.predict,
		name = 'predict'
	)

	train_error = theano.function(
		[],
		T.sqrt(cost),
		givens = {
			x: complete_set,
			xnm: missing_entries
		},
		name = 'train_error'
	)

	# print '... building test error function ...'
	# test_error = theano.function(
	# 	[],
	# 	T.sqrt(cost),
	# 	givens = {
	# 		x: test_set_x,
	# 		xnm: test_not_miss
	# 	},
	# 	name = 'test_error'
	# )

	start_time = timeit.default_timer()

	for epoch in xrange(training_epochs):	
		train_mf()
		print 'Training epoch %d, train error ' % epoch,\
			train_error()
			#, ', test error ', test_error()
		# print 'W: ', da.W.get_value()

	end_time = timeit.default_timer()

	training_time = end_time - start_time

	return predict(dataset)

if __name__ == '__main__':
	# add test?
	print('Test \n')


























