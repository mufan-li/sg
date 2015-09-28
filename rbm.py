
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

'''
#############
# RBM class #
#############
'''

class RBM(object):
	def __init__(
		self,
		input=None,
		input_nm=None,
		n_visible=784,
		n_hidden=500,
		W=None,
		hbias=None,
		vbias=None,
		numpy_rng=None,
		theano_rng=None
	):
		self.n_visible = n_visible
		self.n_hidden = n_hidden

		if numpy_rng is None:
			# create a numpy generator
			numpy_rng = rd.RandomState(1234)

		if theano_rng is None:
			theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

		# need to create shared variables
		if W is None:
			initial_W = np.asarray(
				numpy_rng.uniform(
					low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
					high=4 * np.sqrt(6. / (n_hidden + n_visible)),
					size=(n_visible, n_hidden)
				),
				dtype = theano.config.floatX
			)
			W = theano.shared(value=initial_W, name='W', borrow=True)

		if hbias is None:
			hbias = theano.shared(
				value=np.zeros(
					(1,n_hidden),
					dtype=theano.config.floatX,
				),
				name='hbias',
				borrow=True,
				broadcastable=[True,False]
			)

		if vbias is None:
			vbias = theano.shared(
				value=np.zeros(
					(1,n_visible),
					dtype=theano.config.floatX
				),
				name='vbias',
				borrow=True,
				broadcastable=[True,False]
			)

		self.input = input
		if not input:
			self.input = T.matrix('input')
		self.input_nm = input_nm
		if not input_nm:
			self.input = T.matrix('input_nm')

		self.W = W
		self.hbias = hbias
		self.vbias = vbias
		self.theano_rng = theano_rng
		self.params = [self.W, self.hbias, self.vbias]

	def free_energy(self, v_sample):
		# F = log sum_h exp(-E(v,h))
		wx_b = T.dot(v_sample, self.W) + self.hbias
		vbias_term = T.dot(v_sample, self.vbias.T)
		hidden_term = T.log(1 + T.exp(wx_b))
		return (-vbias_term-hidden_term)

	def propup(self, vis):
		pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
		return [pre_sigmoid_activation, \
				T.nnet.sigmoid(pre_sigmoid_activation)]

	def sample_h_given_v(self, v0_sample):
		pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
		h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
					n=1, p=h1_mean, dtype=theano.config.floatX)
		return [pre_sigmoid_h1, h1_mean, h1_sample]

	def propdown(self, hid):
		pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
		return [pre_sigmoid_activation, \
				T.nnet.sigmoid(pre_sigmoid_activation)]

	def sample_v_given_h(self, h0_sample):
		pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
		v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
					n=1, p=v1_mean, dtype=theano.config.floatX)
		return [pre_sigmoid_v1, v1_mean, v1_sample]

	def gibbs_hvh(self, h0_sample):
		pre_sigmoid_v1, v1_mean, v1_sample = \
			self.sample_v_given_h(h0_sample)
		pre_sigmoid_h1, h1_mean, h1_sample = \
			self.sample_h_given_v(v1_sample)
		return [pre_sigmoid_v1, v1_mean, v1_sample,
				pre_sigmoid_h1, h1_mean, h1_sample]

	def gibbs_vhv(self, v0_sample):
		pre_sigmoid_h1, h1_mean, h1_sample = \
			self.sample_h_given_v(v0_sample)
		pre_sigmoid_v1, v1_mean, v1_sample = \
			self.sample_v_given_h(h1_sample)
		return [pre_sigmoid_h1, h1_mean, h1_sample,
				pre_sigmoid_v1, v1_mean, v1_sample]

	# for scan
	def each_grad(self, x, xnm, chain_end):
		# still need to multiple xnm
		# print cost.ndim
		costs = self.free_energy(x) - self.free_energy(chain_end)
		cost = T.sum(costs)

		gW = (T.grad(cost, self.W, \
						consider_constant=[chain_end]).T * xnm).T
		ghbias = T.grad(cost, self.hbias, \
					consider_constant=[chain_end])
		gvbias = (T.grad(cost, self.vbias, \
						consider_constant=[chain_end]) * xnm)
		return gW, ghbias, gvbias

	def get_cost_updates(self, lr=0.1, persistent=None, k=1):
		pre_sigmoid_ph, ph_mean, ph_sample = \
			self.sample_h_given_v(self.input*self.input_nm)

		if persistent is None:
			chain_start = ph_sample
		else:
			chain_start = persistent

		(
			[
				pre_sigmoid_nvs,
				nv_means,
				nv_samples,
				pre_sigmoid_nhs,
				nh_means,
				nh_samples
			],
			updates
		) = theano.scan(
				self.gibbs_hvh,
				outputs_info = [None, None, None, None, None, \
								chain_start],
				n_steps = k
		)

		chain_end = nv_samples[-1]

		# calc within scan
		# costs = self.free_energy(self.input) -\
		# 			self.free_energy(chain_end)

		# return all the gradients
		(
			[gW_vals, ghbias_vals, gvbias_vals],
			updates2
		) = theano.scan(
				self.each_grad,
				outputs_info = None,
				sequences = [self.input, self.input_nm, chain_end]
		)

		gW = T.sum(gW_vals,0)
		ghbias = T.sum(ghbias_vals,0)
		gvbias = T.sum(gvbias_vals,0)
		print self.vbias.broadcastable
		# gvbias = T.patternbroadcast(gvbias,self.vbias.broadcastable)
		print gvbias.broadcastable

		updates[self.W] = self.W - gW * \
						T.cast(lr, dtype=theano.config.floatX)
		updates[self.vbias] = self.vbias - \
							gvbias * \
							T.cast(lr, dtype=theano.config.floatX)
		updates[self.hbias] = self.hbias - ghbias * \
							T.cast(lr, dtype=theano.config.floatX)

		if persistent:
			updates[persistent] = nh_samples[-1]
			monitoring_cost = self.get_pseudo_likelihood_cost(updates)
		else:
			monitoring_cost = self.get_reconstruction_cost(\
								updates,pre_sigmoid_nvs[-1])

		return monitoring_cost, updates


	def get_pseudo_likelihood_cost(self, updates):
		bit_i_idx = theano.shared(value=0, name='bit_i_idx')
		
		xi = T.round(self.input)
		fe_xi = self.free_energy(xi)

		xi_flip = T.set_subtensor(xi[:, bit_i_idx], \
						1 - xi[:, bit_i_idx])

		fe_xi_flip = self.free_energy(xi_flip)

		cost = T.mean(self.n_visible * \
				T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)) )

		updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

		return cost

	def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
		cross_entropy = T.mean(
			T.sum(
				self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) + 
				(1 - self.input) * \
					T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
				axis = 1
			)
		)

		return cross_entropy

'''
#####################
# run RBM on grades #
#####################
'''

def run_rbm(dataset, learning_rate = 0.1, training_epochs=1,
			batch_size=10,
			n_chains=15, n_samples=10, output_folder='sg_output',
			n_hidden=20):
	# ~80% of data for training
	train_idx = dataset.shape[0]*4/5
	train_set_x = shared_data(dataset[:train_idx, :])
	train_not_miss = shared_data( dataset[:train_idx, :]==0 )
	test_set_x = shared_data(dataset[train_idx:, :])
	test_not_miss = shared_data( dataset[train_idx:, :]==0 )

	n_train_batches = train_set_x.get_value(borrow=True).shape[0]\
			/ batch_size

	index = T.lscalar()
	x = T.matrix('x')
	xnm = T.matrix('xnm') # for not_miss matrix

	rng = numpy.random.RandomState(123)
	theano_rng = RandomStreams(rng.randint(2 ** 30))

	persistent_chain = theano.shared(np.zeros((batch_size, n_hidden),
							dtype=theano.config.floatX),
						borrow=True)

	rbm = RBM(input=x, input_nm=xnm, n_visible= dataset.shape[1],
			n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

	cost, updates = rbm.get_cost_updates(lr=learning_rate,
						persistent=persistent_chain, k=15)

	# create directory
	if not os.path.isdir(output_folder):
		os.makedirs(output_folder)
	os.chdir(output_folder)

	# training
	train_rbm = theano.function(
		[index],
		cost,
		updates = updates,
		givens = {
			x: train_set_x[index* batch_size : (index+1)* batch_size],
			xnm: train_not_miss[index* batch_size:(index+1)*batch_size]
		},
		name = 'train_rbm'
	)

	plotting_time=0.
	start_time = timeit.default_timer()

	# loop through epochs
	for epoch in xrange(training_epochs):
		# loop through training set
		mean_cost = []
		for batch_index in xrange(n_train_batches):
			mean_cost += [train_rbm(batch_index)]

		print 'Training epoch %d, cost is ' % epoch, np.mean(mean_cost)

	end_time = timeit.default_timer()
	pretraining_time = (end_time - start_time)

	print ('Training took %f minutes' % (pretraining_time / 60.))

if __name__ == '__main__':
	test_rbm()




