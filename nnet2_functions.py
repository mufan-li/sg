# functions for nnet2 implementation using theano

import numpy as np
import numpy.random as rd
import theano

# initialize weights matrix
def w_init(n_in, n_out):
	w = rd.uniform(-0.1,0.1,size = (n_in, n_out))
	return np.asarray(w, dtype = theano.config.floatX)

# initialize bias vector
def b_init(n_out):
	b = rd.uniform(-0.1,0.1,n_out)
	return np.asarray(b, dtype = theano.config.floatX)