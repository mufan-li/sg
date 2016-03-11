import timeit
try:
	import PIL.Image as Image
except ImportError:
	import Image

import numpy as np
import numpy.random as rd

import theano
import theano.tensor as T
import os

from theano.tensor.shared_randomstreams import RandomStreams

from sg_functions import *

class dA(object):
	'''Denoising Auto-Encoder class
	'''
	def __init__(
		self,
		numpy_rng,
		theano_rng = None,
		input = None,
		input_nm = None,
		n_visible = 784,
		n_hidden = 500,
		W = None,
		bhid = None,
		bvis = None,
		actv_fcn = None
	):

		self.n_visible = n_visible
		self.n_hidden = n_hidden

		if not theano_rng:
			theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

		if not W:
			initial_W = np.asarray(
				numpy_rng.uniform(
					low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
					high=4 * np.sqrt(6. / (n_hidden + n_visible)),
					size = (n_visible, n_hidden)
				),
				dtype = theano.config.floatX
			)
			W = theano.shared(value = initial_W, name = 'W',
					borrow = True)

		if not bvis:
			bvis = theano.shared(
				value = np.zeros(
					n_visible,
					dtype = theano.config.floatX
				),
				borrow = True
			)

		if not bhid:
			bhid = theano.shared(
				value = np.zeros(
					n_hidden,
					dtype = theano.config.floatX
				),
				name = 'b',
				borrow = True
			)

		self.W = W
		self.b = bhid
		self.b_prime = bvis
		self.W_prime = self.W.T
		self.theano_rng = theano_rng

		if input is None:
			self.x = T.dmatrix(name = 'input')
		else:
			self.x = input
		self.xnm = input_nm

		self.params = [self.W, self.b, self.b_prime]

		if actv_fcn is None:
			actv_fcn = T.nnet.sigmoid
		self.actv_fcn = actv_fcn

	def get_corrupted_input(self, input, corruption_level):
		return self.theano_rng.binomial(size = input.shape, n = 1,
				p = 1 - corruption_level,
				dtype = theano.config.floatX) * input

	def get_hidden_values(self, input):
		return self.actv_fcn(T.dot(input, self.W) + 
				self.b)

	def get_reconstructed_input(self, hidden):
		return self.actv_fcn(T.dot(hidden, self.W_prime)+
				self.b_prime)
	
	def predict(self, input):
		hidden = self.get_hidden_values(input)
		return self.get_reconstructed_input(hidden)

	def get_cost(self, x, xnm, corruption_level):
		tilde_x = self.get_corrupted_input(x, corruption_level)
		y = self.get_hidden_values(tilde_x)
		z = self.get_reconstructed_input(y)
		# L = - T.sum(self.x * T.log(z) * self.xnm + 
		# 		(1 - self.x) * T.log(1 - z) * self.xnm,
		# 		axis=1)
		# cost = T.mean(L)

		# use squared error instead
		cost = T.sum(T.square( x - z * xnm ))/T.sum(xnm)
		return cost

		# for scan
	def each_grad(self, x, xnm, corruption_level):
		cost = self.get_cost(x, xnm, corruption_level)

		gW = (T.grad(cost, self.W).T * xnm).T
		gb = T.grad(cost, self.b)
		gbp = (T.grad(cost, self.b_prime) * xnm)
		return gW, gb, gbp

	def get_cost_updates(self, corruption_level, 
			learning_rate, momentum_const):

		# return all the gradients
		(
			[gW_vals, gb_vals, gbp_vals],
			updates # placeholder
		) = theano.scan(
				self.each_grad,
				outputs_info = None,
				sequences = [self.x, self.xnm],
				non_sequences = corruption_level
		)

		# gparams = T.grad(cost, self.params)
		gparams = [T.mean(gW_vals,0), T.mean(gb_vals,0), \
				T.mean(gbp_vals,0)]
		vparams = [theano.shared(np.zeros(param.get_value().shape),
						borrow=True,
						broadcastable=param.broadcastable)
						for param in self.params]

		# momentum
		update1 = [
			(vparam, T.cast(momentum_const, 
				dtype=theano.config.floatX) * vparam \
					+ T.cast(learning_rate, 
						dtype=theano.config.floatX) * gparam)
			for vparam, gparam in zip(vparams, gparams)
		]
		# change
		update2 = [
			(param, param - vparam) 
			for param, vparam in zip(self.params, vparams)
		]
		updates += update1 + update2

		cost = self.get_cost(self.x, self.xnm, corruption_level)

		# updates = [
		# 	(param, param - learning_rate * gparam)
		# 	for param, gparam in zip(self.params, gparams)
		# ]

		return (cost, updates)

def run_dA(dataset, learning_rate = 0.1, training_epochs = 15,
		batch_size = 20, n_hidden = 100, corruption_level = 0.3,
		momentum_const = 0.9, actv_fcn = None):
	# input.nm is used in gradients
	theano.config.on_unused_input = 'warn'

	# ~80% of data for training
	train_idx = dataset.shape[0]*4/5
	train_set_x = shared_data(dataset[:train_idx, :])
	train_not_miss = shared_data( ~(dataset[:train_idx, :]==0) )
	test_set_x = shared_data(dataset[train_idx:, :])
	test_not_miss = shared_data( ~(dataset[train_idx:, :]==0) )

	# complete set
	missing_entries = shared_data(~(dataset==0))
	complete_set = shared_data(dataset)

	n_train_batches = train_set_x.get_value(borrow=True).shape[0]\
			/ batch_size

	index = T.lscalar()
	index_end = T.lscalar()
	x = T.matrix('x')
	xnm = T.matrix('xnm') # for not_miss matrix

	rng = numpy.random.RandomState(123)
	theano_rng = RandomStreams(rng.randint(2 ** 30))

	da = dA(
		numpy_rng = rng,
		theano_rng = theano_rng,
		input = x,
		input_nm = xnm,
		n_visible = dataset.shape[1],
		n_hidden = n_hidden,
		actv_fcn = actv_fcn
	)

	cost, updates = da.get_cost_updates(
		corruption_level = corruption_level,
		learning_rate = learning_rate,
		momentum_const = momentum_const
	)

	print '... building training function ...'
	train_da = theano.function(
		[index],
		cost,
		updates = updates,
		givens = {
			x: train_set_x[index * batch_size: (index+1) * batch_size],
			xnm: train_not_miss[index*batch_size:(index+1)*batch_size]
		},
		name = 'train_da'
	)

	print '... building predict function ...'
	predict = theano.function(
		[x],
		da.predict(x),
		name = 'predict'
	)

	print '... building training error function ...'
	error, _ = da.get_cost_updates(
		corruption_level = 0,
		learning_rate = learning_rate,
		momentum_const = momentum_const
	)
	
	train_error = theano.function(
		[],
		T.sqrt(error),
		givens = {
			x: train_set_x,
			xnm: train_not_miss
		},
		name = 'train_error'
	)

	print '... building test error function ...'
	test_error = theano.function(
		[],
		T.sqrt(error),
		givens = {
			x: test_set_x,
			xnm: test_not_miss
		},
		name = 'test_error'
	)

	get_hid = theano.function(
		[],
		da.get_hidden_values(x),
		givens = {
			x: complete_set
		}
	)

	start_time = timeit.default_timer()

	for epoch in xrange(training_epochs):
		for batch_index in xrange(n_train_batches):
			train_da(batch_index)

			# if (batch_index % (n_train_batches/10) == 0):
		print 'Training epoch %d, train error ' % epoch,\
			train_error(), ', test error ', test_error()
				# print np.linalg.norm(da.W.get_value()),\
				# 	np.linalg.norm(da.b.get_value()),\
				# 	np.linalg.norm(da.b_prime.get_value())
		# print 'W: ', da.W.get_value()

	end_time = timeit.default_timer()

	training_time = end_time - start_time

	return predict(dataset), get_hid()

if __name__ == '__main__':
	dataset = np.asarray( [[0,0.5],[1,0],[0.5,1],[0.6,0.1],[0,0.9]] )
	run_dA(dataset, learning_rate = 0.1, training_epochs = 15,
		batch_size = 1, n_hidden = 1, corruption_level = 0.3)	
























