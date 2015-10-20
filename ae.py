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
		bvis = None
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

	def get_corrupted_input(self, input, corruption_level):
		return self.theano_rng.binomial(size = input.shape, n = 1,
				p = 1 - corruption_level,
				dtype = theano.config.floatX) * input

	def get_hidden_values(self, input, input_nm):
		return T.nnet.sigmoid(T.dot(input * input_nm, self.W) + 
				self.b)

	def get_reconstructed_input(self, hidden):
		return T.nnet.sigmoid(T.dot(hidden, self.W_prime)+
				self.b_prime)

	def get_cost_updates(self, corruption_level, learning_rate):
		tilde_x = self.get_corrupted_input(self.x, corruption_level)
		y = self.get_hidden_values(tilde_x, self.xnm)
		z = self.get_reconstructed_input(y)

		# L = - T.sum(self.x * T.log(z) * self.xnm + 
		# 		(1 - self.x) * T.log(1 - z) * self.xnm,
		# 		axis=1)
		# cost = T.mean(L)

		# use squared error instead
		cost = T.sum(T.square( self.x - z * self.xnm ))/T.sum(self.xnm)

		gparams = T.grad(cost, self.params)

		updates = [
			(param, param - learning_rate * gparam)
			for param, gparam in zip(self.params, gparams)
		]

		return (cost, updates)

def run_dA(dataset, learning_rate = 0.1, training_epochs = 15,
		batch_size = 20, n_hidden = 100, corruption_level = 0.3):
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
		n_hidden = n_hidden
	)

	cost, updates = da.get_cost_updates(
		corruption_level = corruption_level,
		learning_rate = learning_rate
	)

	print '... building training function ...'
	train_da = theano.function(
		[index],
		cost,
		updates = updates,
		givens = {
			x: train_set_x[index * batch_size: (index+1) * batch_size],
			xnm: train_not_miss[index*batch_size:(index+1)*batch_size]
		}
	)

	start_time = timeit.default_timer()

	for epoch in xrange(training_epochs):
		c = []
		for batch_index in xrange(n_train_batches):
			c.append(train_da(batch_index))
		print 'Training epoch %d, cost ' % epoch, np.mean(c)
		# print 'W: ', da.W.get_value()

	end_time = timeit.default_timer()

	training_time = end_time - start_time

if __name__ == '__main__':
	dataset = np.asarray( [[0,0.5],[1,0],[0.5,1]] )
	run_dA(dataset, learning_rate = 0.1, training_epochs = 15,
		batch_size = 1, n_hidden = 10, corruption_level = 0.3)	
























