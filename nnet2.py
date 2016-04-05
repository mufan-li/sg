# new neural network class defined using theano

import numpy as np
import numpy.random as rd
import theano.tensor as T
import theano
import timeit
import lasagne as lsg

from nnet2_functions import *
from sg_functions import *
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.nnet.bn import batch_normalization

class nnet_layer(object):
	""" the most basic basic neural net layer class with theano
	"""

	# x - input data
	# n_in - # of nodes in the previous layer
	# n_out - # of nodes in the current layer
	# layer - the layer count so far, for symbol difference
	# act - activivation function for this layer
	# dropout_rate - probability of dropping an input node
	# dropout_on - Theano tensor, switch for dropout
	def __init__(self, x, n_in, n_out, dropout_on,
				layer=0, act=T.nnet.sigmoid,
				w = None, b = None, dropout_rate=0.3):
		if w==None:
			w = theano.shared(
				value=w_init(n_in, n_out),
				name='w'+str(layer),
				borrow=True
			)

		if b==None:
			b = theano.shared(
				value=b_init(n_out),
				name='b'+str(layer),
				borrow=True
			)

		self.w = w
		self.b = b
		self.gamma = theano.shared(value = numpy.ones((n_out,), 
						dtype=theano.config.floatX), name='gamma')
		self.beta = theano.shared(value = numpy.zeros((n_out,), 
						dtype=theano.config.floatX), name='beta')

		rng = np.random.RandomState(42)
		srng = RandomStreams(rng.randint(10**9))
		mask = srng.binomial(n=1, p=1-dropout_rate, size=x.shape)
		cast_mark = T.cast(mask, theano.config.floatX)

		drop_input = T.switch(dropout_on, x*cast_mark,x*(1-dropout_rate))
		lin_output = T.dot(drop_input, self.w) + self.b

		bn_output = batch_normalization(inputs = lin_output,
			gamma = self.gamma, beta = self.beta, 
			mean = lin_output.mean((0,), keepdims=True),
			std = lin_output.std((0,), keepdims = True),
						mode='low_mem')

		self.output = (
			bn_output if act is None
			else act(bn_output)
		)

		self.params = [self.w, self.b]

class nnet2(object):
	""" the entire neural net implemented with theano
	"""

	def __init__(self, x, n_in, v_hidden, n_out, dropout_on,
		hid_act = relu, out_act = T.nnet.softmax,
		dropout_rate = 0.3):

		if type(v_hidden) != type([0]) and \
			type(v_hidden) != type(np.array([])):
			v_hidden = [v_hidden]

		self.n_in = n_in
		self.v_hidden = v_hidden
		self.n_out = n_out

		n_hid = len(v_hidden)
		layers_in = np.concatenate(([n_in],v_hidden))
		layers_out = np.concatenate((v_hidden,[n_out]))
		layers_act = [hid_act] * n_hid + [out_act]

		self.layers = []
		self.params = []
		x_in = x

		for i in range(n_hid+1):
			self.layers.append(
				nnet_layer(x=x_in, 
					n_in=layers_in[i],
					n_out=layers_out[i],
					dropout_on = dropout_on,
					layer=i, 
					act=layers_act[i],
					dropout_rate = dropout_rate*int(i>0))
				)
			self.params += self.layers[i].params
			x_in = self.layers[i].output

		# output of the final layer
		self.output = x_in
		self.outclass = T.argmax(self.output, axis=1)

	def mse(self, y):
		return T.mean(T.square(self.output - y))

	def nll(self, y):
		# return - T.mean(T.dot(T.log(self.output.T), y))
		return -T.mean(
			T.log(self.output)[T.nonzero(y)]
			)

	def nll2(self, y):
		# for predicting whether a course is taken
		return -T.mean(
			T.log(self.output)[T.nonzero(y)] + 
			T.log(1 - self.output)[T.nonzero(1 - y)]
			)

	def error(self,y):
		return T.mean(T.neq(self.outclass, T.argmax(y, axis=1)))

	def error2(self,y):
		# for predicting whether a course is taken
		return T.mean(
				T.neq(T.round(self.output),y)
				)

def run_nnet(dataset, labelset, learning_rate = 1e-5, 
	training_epochs = 15,
	batch_size = 20, v_hidden = [100, 100],
	momentum_const = 0.9, cost_type = 'MSE',
	actv_fcn = None, out_actv_fcn = None,
	dropout_rate = 0.3, dropout_switch = True,
	lr_decay = 1e-2, pred_course = False,
	update_method = 'momentum'):
	# input.nm is used in gradients
	theano.config.on_unused_input = 'warn'

	# ~80% of data for training
	train_idx = dataset.shape[0]*4/5
	idx_array = np.arange(dataset.shape[0])
	# Shuffle indices
	rd.shuffle(idx_array)
	train_set_x = shared_data(dataset[idx_array[:train_idx], :])
	# train_not_miss = shared_data( ~(dataset[:train_idx, :]==0) )
	test_set_x = shared_data(dataset[idx_array[train_idx:], :])
	# test_not_miss = shared_data( ~(dataset[train_idx:, :]==0) )

	train_set_y = shared_data(labelset[idx_array[:train_idx], :])
	test_set_y = shared_data(labelset[idx_array[train_idx:], :])

	# complete set
	# missing_entries = shared_data(~(dataset==0))
	complete_set = shared_data(dataset)

	n_train_batches = train_set_x.get_value(borrow=True).shape[0]\
			/ batch_size

	index = T.lscalar()
	index_end = T.lscalar()
	lr = T.scalar('lr')
	x = T.matrix('x')
	y = T.matrix('y')
	dropout_on = T.scalar('dropout_on')
	# xnm = T.matrix('xnm') # for not_miss matrix

	rng = numpy.random.RandomState(123)
	theano_rng = RandomStreams(rng.randint(2 ** 30))

	if actv_fcn == None:
		actv_fcn = T.nnet.sigmoid
		actv_fcn_name = 'Sigmoid'
	else:
		actv_fcn_name = actv_fcn.func_name
	
	if out_actv_fcn == None:
		out_actv_fcn = T.nnet.softmax

	nn = nnet2(x, dataset.shape[1], v_hidden, labelset.shape[1], 
		dropout_on = dropout_on, dropout_rate = dropout_rate,
		hid_act = actv_fcn, out_act = out_actv_fcn)
	output = nn.output

	if (cost_type == 'NLL') & (~pred_course):
		cost = nn.nll(y)
	elif (cost_type == 'NLL') & (pred_course):
		cost = nn.nll2(y)
	else:
		cost = nn.mse(y)

	# cost2 = T.log(nn.output)[T.arange(y.shape[0]), T.argmax(y,axis=1)]

	if pred_course:
		error_rate = nn.error2(y)
	else:
		error_rate = nn.error(y)

	if (update_method == 'adam'):
		updates = lsg.updates.adam(
			cost, 
			nn.params, 
			learning_rate=lr, 
			beta1=0.9, 
			beta2=0.999, 
			epsilon=1e-08)
	elif (update_method == 'adadelta'):
		 updates = lsg.updates.adadelta(
		 	cost,
		 	nn.params,
		 	learning_rate=lr, 
		 	rho=0.95, 
		 	epsilon=1e-06)
	else:
		gparams = [T.grad(cost, param) for param in nn.params]
		
		vparams = [theano.shared(param.get_value(),borrow=True)
					for param in nn.params
		]

		update1 = [
			(vparam, momentum_const * vparam - lr * gparam) 
			for vparam, gparam in zip(vparams, gparams)
		]
		update2 = [
			(param, param + vparam) 
			for param, vparam in zip(nn.params, vparams)
		]
		updates = update1 + update2

	

	print '... building training function ...'
	train_model = theano.function(
		inputs=[index, dropout_on, lr],
		outputs=[cost],
		updates=updates,
		givens={
			x: train_set_x[index * batch_size: (index+1) * batch_size],
			y: train_set_y[index * batch_size: (index+1) * batch_size]
		}
	)

	print '... building predict function ...'
	predict = theano.function(
		[x,dropout_on],
		output,
		name = 'predict',
		allow_input_downcast = True
	)
	
	print '... building training error function ...'
	train_error = theano.function(
		inputs = [dropout_on],
		outputs = [cost, error_rate],
		givens = {
			x: train_set_x,
			y: train_set_y
		},
		name = 'train_error'
	)

	print '... building test error function ...'
	test_error = theano.function(
		inputs = [dropout_on],
		outputs = [cost, error_rate],
		givens = {
			x: test_set_x,
			y: test_set_y
		},
		name = 'test_error'
	)

	start_time = timeit.default_timer()
	train_MSE = np.zeros(training_epochs)
	train_error_rate = np.zeros(training_epochs)
	test_MSE = np.zeros(training_epochs)
	test_error_rate = np.zeros(training_epochs)

	import matplotlib.pyplot as plt

	for epoch in xrange(training_epochs):
		cur_lr = learning_rate * (1-lr_decay)**epoch

		for batch_index in xrange(n_train_batches):	
			train_model(batch_index, 1, cur_lr)
		
		train_MSE[epoch], train_error_rate[epoch] = train_error(0)
		test_MSE[epoch], test_error_rate[epoch] = test_error(0)
		
		if ((epoch % 10) == 0):
			print 'v_hid: ', v_hidden, \
				', batch: ', batch_size, \
				', actv_fcn: ', actv_fcn_name
			print 'lr: ', learning_rate, \
				', decay: ', lr_decay, \
				', momentum: ', momentum_const, \
				', dropout: ', dropout_rate
		print 'Epoch ',epoch,', train ', cost_type,' ',\
			np.round(train_MSE[epoch],4), ', train error ', \
			np.round(train_error_rate[epoch],4),\
			', test error ', np.round(test_error_rate[epoch],4)

	end_time = timeit.default_timer()

	training_time = end_time - start_time

	return predict(dataset, 0), train_MSE, \
		test_MSE, train_error_rate, test_error_rate


if __name__ == "__main__":

	sN = 100
	learning_rate = 1e-5
	momentum = 0.9

	xval = rd.uniform(0,5,sN).reshape(sN,1)
	trainx = theano.shared(
				value = np.matrix(xval,
					dtype = theano.config.floatX),
				borrow = True
				)

	yval = np.asarray((xval>2).astype(float),
					dtype = theano.config.floatX)
	yval = np.concatenate([yval,1-yval],1);
	trainy = theano.shared(
			value = yval,
			borrow = True
			)

	x = T.matrix('x')
	y = T.matrix('y')
	end = T.lscalar()

	nn = nnet2(x, xval.shape[1], 2, yval.shape[1], 
		out_act = T.nnet.softmax)
	output = nn.output
	# cost = nn.mse(y)
	cost = nn.nll(y)
	gparams = [T.grad(cost, param) for param in nn.params]
	
	vparams = [theano.shared(param.get_value(),borrow=True)
				for param in nn.params
	]

	update1 = [
		(vparam, momentum * vparam - learning_rate * gparam) 
		for vparam, gparam in zip(vparams, gparams)
	]
	update2 = [
		(param, param + vparam) 
		for param, vparam in zip(nn.params, vparams)
	]

	train_model = theano.function(
		inputs=[end],
		outputs=[cost],
		updates=update1 + update2,
		givens={
			x: trainx[:end],
			y: trainy[:end]
		}
	)

	bN = sN
	for i in range(10):
		print train_model(bN)[0]

	# theano.printing.pprint(nn.output)
	# theano.printing.debugprint(gparams[0])




