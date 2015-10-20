import theano
import theano.tensor as T
import numpy as np
from rbm import *
from ae import *

def test_ae(learning_rate = 0.1, training_epochs = 1,
	batch_size = 1, n_hidden=5, corruption_level = 0.3):
	dataset = np.asarray( [[0,0.5],[1,0],[0.5,1]] )
	train_set_x = shared_data(dataset)
	train_not_miss = shared_data(np.asarray( [[0,1],[1,0],[1,1]] ))

	index = T.lscalar()
	index2 = T.lscalar()
	x = T.matrix('x')
	xnm = T.matrix('xnm')

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

	print 'W: ', da.W.get_value()
	
	crpt = da.get_corrupted_input(da.x, corruption_level)
	crpt_fn = theano.function(
		[],
		crpt,
		givens = {
			x: train_set_x
		},
		name = 'crpt_fn'
	)
	print 'Corrupt x: ', crpt_fn()

	hid = da.get_hidden_values(da.x, da.xnm)
	hid_fn = theano.function(
		[],
		hid,
		givens = {
			x: train_set_x,
			xnm: train_not_miss
		},
		name = 'hid_fn'
	)
	print 'h: ', hid_fn()

	recon = da.get_reconstructed_input(hid)
	recon_fn = theano.function(
		[],
		recon,
		givens = {
			x: train_set_x,
			xnm: train_not_miss
		},
		name = 'recon_fn'
	)
	print 'z: ', recon_fn()

	cost = T.sum(T.square(da.x - recon * da.xnm))/T.sum(da.xnm)
	cost_fn = theano.function(
		[],
		cost,
		givens = {
			x: train_set_x,
			xnm: train_not_miss
		},
		name = 'cost_fn'
	)
	print 'cost: ', cost_fn()

	gp = T.grad(cost, da.params)
	gp_fn = theano.function(
		[],
		gp,
		givens = {
			x: train_set_x,
			xnm: train_set_x
		},
		name = 'gp_fn'
	)
	print 'grad: ', gp_fn()

	updates = [
		(param, param - learning_rate * gparam)
		for param, gparam in zip(da.params, gp)
	]
	train_da = theano.function(
		[],
		cost,
		updates = updates,
		givens = {
			x: train_set_x,
			xnm: train_not_miss
		},
		name = 'train_da'
	)
	train_da()
	print 'updated W:', da.W.get_value()

def test_rbm2(learning_rate = 0.1, training_epochs=1,
			n_chains=15, n_samples=10, output_folder='sg_output',
			n_hidden=1):
	#
	dataset = np.asarray( [[0,0.5],[1,0],[0.5,1]] )
	train_set_x = shared_data(dataset)
	train_not_miss = shared_data(np.asarray( [[0,1],[1,0],[1,1]] ))

	index = T.lscalar()
	index2 = T.lscalar()
	x = T.matrix('x')
	xnm = T.matrix('xnm')

	rng = numpy.random.RandomState(123)
	theano_rng = RandomStreams(rng.randint(2 ** 30))

	rbm = gbRBM(input=x, input_nm=xnm, n_visible= dataset.shape[1],
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

	h_mean = theano.function(
		[index,index2],
		rbm.h_mean,
		givens = {
			x: train_set_x[index: index2]
		}
	)

	predict = theano.function(
		[index, index2],
		rbm.v_mean,
		givens = {
			x: train_set_x[index: index2]
		},
		name = 'predict'
	)

	rmse = theano.function(
		[],
		T.sqrt(T.sum(T.square((rbm.v_mean - x)*xnm))/T.sum(xnm)),
		givens = {
			x: train_set_x,
			xnm: train_not_miss
		},
		name = 'sqr'
	)

	print 'grad W vector ', gW_fn2(0,3)
	print 'grad W vector sum ', np.sum(gW_fn2(0,3),0)

	print 'vb ', rbm.vbias.get_value()
	print 'grad vb vector ', gv_fn(0,3)
	print 'grad vb vector sum ', np.sum(gv_fn(0,3),0)

	print 'hb ', rbm.hbias.get_value()
	print 'grad hb vector ', gh_fn(0,3)
	print 'grad hb vector sum ', np.sum(gh_fn(0,3),0)

	print 'h_mean ', np.asarray(h_mean(0,3))
	data_predict = np.asarray(predict(0,3))
	print 'predict ', data_predict, data_predict.shape

	print 'RMSE ', rmse()

def test_rbm_gb(learning_rate = 0.1, training_epochs=1,
			n_chains=15, n_samples=10, output_folder='sg_output',
			n_hidden=1):
	#
	dataset = np.asarray( [[0,0.5],[1,0],[0.5,1]] )
	train_set_x = shared_data(dataset)
	train_not_miss = shared_data(np.asarray( [[0,1],[1,0],[1,1]] ))

	index = T.lscalar()
	index2 = T.lscalar()
	x = T.matrix('x')
	xnm = T.matrix('xnm')

	rng = numpy.random.RandomState(123)
	theano_rng = RandomStreams(rng.randint(2 ** 30))

	rbm = gbRBM(input=x, input_nm=xnm, n_visible= dataset.shape[1],
			n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng,
			sgm = 0.2)

	pre_sigmoid_ph, ph_mean, ph_sample = \
		rbm.sample_h_given_v(x)
	pre_sigmoid_pv, pv_mean, pv_sample = \
		rbm.sample_v_given_h(ph_sample)
	
	test_h_sample = theano.function(
		[],
		[ph_mean, ph_sample],
		givens = {
		x: train_set_x
		},
		name = 'test_h_sample'
	)
	test_v_sample = theano.function(
		[],
		[pv_mean, pv_sample],
		givens = {
		x: train_set_x
		},
		name = 'test_v_sample'
	)
	print 'h_sample: ', test_h_sample()
	print 'v_sample: ', test_v_sample()

		# predicting function
	print '... building predict function'
	predict = theano.function(
		[],
		rbm.v_mean,
		givens = {
			x: train_set_x
		},
		name = 'predict'
	)
	print 'predict: ', predict()

	fe = rbm.free_energy(x)
	fe_fn = theano.function(
		[],
		fe,
		givens = {
		x: train_set_x
		},
		name = 'fe_fn'
	)
	print 'fe: ', fe_fn()

if __name__ == '__main__':
	# test_rbm_gb()
	# test_rbm2()
	test_ae()










