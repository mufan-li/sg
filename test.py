import theano
import theano.tensor as T
import numpy as np
from rbm import *

# x, w = T.scalars('x','w')
# xs = T.vector('xs')
# # cost = x * w
# # gw = T.grad(cost, w)

# def gen_grad(x,w):
# 	return T.grad(x*w, w)

# # f = theano.function([x,w], gw)

# (
# 	results, updates
# ) = theano.scan(
# 	gen_grad,
# 	outputs_info = None,
# 	sequences = xs,
# 	non_sequences = w
# 	)

# f2 = theano.function([xs,w], results)
# print f2([1,2,3],3)

def test_rbm2(learning_rate = 0.1, training_epochs=1,
			n_chains=15, n_samples=10, output_folder='sg_output',
			n_hidden=1):
	#
	dataset = np.asarray( [[2,3],[1,2],[-1,1]] )
	train_set_x = shared_data(dataset)
	train_not_miss = shared_data(np.asarray( [[0,1],[1,0],[1,1]] ))

	index = T.lscalar()
	index2 = T.lscalar()
	x = T.matrix('x')
	xnm = T.matrix('xnm')

	rng = numpy.random.RandomState(123)
	theano_rng = RandomStreams(rng.randint(2 ** 30))

	rbm = RBM(input=x, input_nm=xnm, n_visible= dataset.shape[1],
			n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

	fe = rbm.free_energy(x)
	fe_fn = theano.function(
		[index,index2],
		fe,
		givens = {
		x: train_set_x[index:index2]
		},
		name = 'fe_fn'
	)

	# CD-1
	pre_sigmoid_ph, ph_mean, ph_sample = \
		rbm.sample_h_given_v(x)
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
			rbm.gibbs_hvh,
			outputs_info = [None, None, None, None, None, \
							ph_sample],
			n_steps = 1
	)

	chain_end = nv_samples[-1]
	fe2 = rbm.free_energy(chain_end)

	fe_fn2 = theano.function(
		[index, index2],
		fe2,
		givens = {
		x: train_set_x[index:index2]
		},
		name = 'fe_fn2'
	)	

	print 'W ', rbm.W.get_value()
	print 'FE of input ' , fe_fn(0,3)
	print 'FE of CD1 ', fe_fn2(0,3)
	print 'costs.flatten() ', (fe_fn(0,3) - fe_fn2(0,3)).flatten()
	
	# find gradient of costs
	costs = (fe - fe2)

	(
		[gW2,gh2,gv2],
		updates
	) = theano.scan(
		rbm.each_grad,
		outputs_info = None,
		sequences = [x, xnm, chain_end]
	)

	gW_fn2 = theano.function(
		[index, index2],
		gW2,
		givens = {
		x: train_set_x[index:index2],
		xnm: train_not_miss[index:index2]
		},
		name = 'gW_fn2'
	)

	gv_fn = theano.function(
		[index, index2],
		gv2,
		givens = {
		x: train_set_x[index:index2],
		xnm: train_not_miss[index:index2]
		},
		name = 'gv_fn'
	)

	gh_fn = theano.function(
		[index, index2],
		gh2,
		givens = {
		x: train_set_x[index:index2],
		xnm: train_not_miss[index:index2]
		},
		name = 'gh_fn'
	)

	print 'grad W vector ', gW_fn2(0,3)
	print 'grad W vector sum ', np.sum(gW_fn2(0,3),0)

	print 'vb ', rbm.vbias.get_value()
	print 'grad vb vector ', gv_fn(0,3)
	print 'grad vb vector sum ', np.sum(gv_fn(0,3),0)

	print 'hb ', rbm.hbias.get_value()
	print 'grad hb vector ', gh_fn(0,3)
	print 'grad hb vector sum ', np.sum(gh_fn(0,3),0)

if __name__ == '__main__':
	test_rbm2()










