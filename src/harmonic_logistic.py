from lasagne.updates import sgd
from theano import tensor as T, function, shared
import numpy as np


def logistic(length, input_matrix=None, init_params=None):
	"""
	Given a symbolic theno variable representing an input matrix, and a shape
	parameter representing the intended dimension of the input vector, this
	function creates a logistic unit, providing a symbolic theano variable
	representing the output activation, and it returns the symbolic shared
	variable representing the weights (parameters).

	The input matrix should be set up so that the first dimension indexes
	individual vectors or "examples" of length ``length``.  The activation of
	the logistic unit will be calculated for each example, so the output is a
	vector of activations.

	Weights are initialized to zeros.
	"""
	# Create the input if it isn't provided
	if input_matrix is None:
		input_matrix = T.dmatrix()

	# Define a weights vector of shared variables.  
	#	- The first entry in the weights vector is a bias, which is added on
	#		to value of the weights dotted with the inputs without the bias.
	if init_params is None:
		params = shared(np.zeros((1,length+1)), broadcastable=(True,False))

	# But if weights were supplied, use them.  Do some validation first though.
	else:
		params = init_params
		expected_shape = (1, length+1)
		if not init_params.get_value().shape == expected_shape:
			raise ValueError(
				'init_params should have shape ``(1, length+1)``, where length '
				'is the first argument supplied to ``logistic()``.  In this '
				'case that should be ``%s`` Instead got ``%s``.' 
				% ( str(expected_shape), str(init_params.get_value().shape))
			)

	# Create the symbolic output activation for the logistic unit
	bias = params[0,0]
	dot_product = T.dot(input_matrix, params[:,1:].T)
	output_vector = sigmoid(dot_product + bias)

	# Return the output as well as the parameter vector
	return input_matrix, params, output_vector


def sigmoid(a):
	"""
	Computes the signmoid activation on input `a`.  This works whether a is a
	number or a symbolic variable.
	"""
	return 1 / (1 + np.exp(-a))


def harmonic_logistic(
	lengths, input_matrix=None, init_params=None
):
	"""
	Creates a theano computation graph as follows:
	 - The input is split into submatrices along axis 1, such that the ``i``th
		submatrix is a batch of rows of length equal ``lengths[i]``.
	 - Each submatrix is then passed through its own logistic regression type
	   calculation.  
	 - The outputs of each logistic regression unit are then combined, by
	   taking their harmonic mean

	Inputs
		``lengths`` determines how the input matrix is split into submatrices.

		``input_matrix`` should be a theano symbolic matrix variable, or None.
		It represents a batch of feature vectors, with each row being one
		feature vector.  If ``input_matrix`` is None, a variable will be made.

		``init_params`` should be a list of theano shared variables to be used
		as the weights in each logistic unit, or None.  Therefore the ``i``th
		shared variable should have shape ``(1, length[i]+1)``.  If
		``init_params`` is None, then the shared variables representing the
		parameters will be created.

	Returns
		``input_matrix`` the theano matrix representing the inputs to the
		computation graph that this function builds.
		
		``params``	a list of theano shared variables representing the
		parameters in the logistic units.

		``logistic_outputs`` a theano matrix representing the outputs from the 
		logistic units 

		``harmonic_output`` a theano vector representing the harmonic mean of
		the outputs from the logistic units (taken along axis 1), which
		represents the overall output of the harmonic-logistic computation
		graph.
	"""

	# If the inputs don't exist make them
	if input_matrix is None:
		input_matrix = T.dmatrix()

	params_supplied = init_params is not None

	# Create a logistic unit around each input vector
	params = []
	logistic_outputs = []
	prev_endpoint = 0
	for i, length in enumerate(lengths):

		# Create a logistic unit on the specified slice of the input matrix
		weights = init_params[i] if params_supplied else None
		input_submatrix, weights, logistic_output = logistic(
			length,
			input_matrix[:,prev_endpoint:prev_endpoint + length],
			weights
		)
		prev_endpoint += length

		logistic_outputs.append(logistic_output)
		params.append(weights)

	# Make a symbolic variable representing the harmonic mean of these outputs
	all_logistic_outputs = T.concatenate(logistic_outputs, axis=1)
	inv = 1/all_logistic_outputs
	summ = T.sum(inv, axis=1)
	harmonic_output = len(lengths)/summ

	# We return the inputs, the parameters, the component logistic outputs, and
	# the overall output.  The reason we return the inputs is because they may
	# have been created in this function (if they weren't passed in).
	return input_matrix, params, logistic_outputs, harmonic_output


class HarmonicLogistic(object):

	def __init__(
		self, lengths=[], input_matrix=None, target_vector=None,
		params=None, learning_rate=1.0, load=None, clip = 0.001,
		progress_factor_threshold=0.05
	):

		"""
		``target_vector`` should be a theano symbolic vector variable, or None.
		It represents the "corect" value associated to every input vector.
		"""

		# Register arguments to instance
		self.lengths = lengths
		self.input_matrix = input_matrix
		self.target_vector = target_vector
		self.params = params
		self.learning_rate = shared(learning_rate)
		self.clip = clip
		self.progress_factor_threshold = progress_factor_threshold

		# If a load path is given, load the params from that path
		if load is not None:
			self._load(load)

		# Make the architecture, then compile a training and prediction
		# function for the architecture.
		self._build_and_compile_model()


	def _build_and_compile_model(self):

		# Make the target vector if none was supplied
		if self.target_vector is None:
			self.target_vector = T.dvector()

		# Make the computation graph.  First, make the harmonic-logistic unit.
		self.input_matrix, self.params, self.logistic_outputs, self.output = (
			harmonic_logistic(self.lengths, self.input_matrix, self.params))

		# Now define the loss function as the sum of squared errors
		self.loss = T.sum((self.output - self.target_vector)**2)

		# Define the stochastic gradient updates
		updates = []
		adjustments = []
		for param in self.params:
			gradient = T.clip(T.grad(self.loss, param), -self.clip, self.clip)
			adjustment = -self.learning_rate * gradient
			new_val = param + adjustment
			updates.append((param, new_val))
			adjustments.append(adjustment)

		#updates = sgd(self.loss, self.params, self.learning_rate)

		# Define the training function
		self._train = function(
			[self.input_matrix, self.target_vector],
			[self.output, self.loss] + adjustments,
			updates=updates
		)
		self._predict = function([self.input_matrix], self.output)


	def train(self, input_matrix, target_vector):
		return_vals = self._train(input_matrix, target_vector)
		return return_vals


	def predict(self, input_matrix):
		outputs = self._predict(input_matrix)
		return outputs


	def get_param_values(self):
		return [p.get_value() for p in self.params]


	def fit(self, input_matrix, target_vector, tolerance=1e-8, verbose=True):
		change_in_loss = None
		previous_loss = None
		prev_params = None

		len_recall = 20

		recall_adjustments = [
			np.zeros((sum(self.lengths) + len(self.lengths)))
		] * len_recall

		i = 0
		converged = False
		while not converged:
			i = (i+1) % len_recall

			returns = self.train(input_matrix, target_vector)
			outputs, loss = returns[:2]
			adjustments = returns[2:]

			change_in_loss = (
				None if previous_loss is None else
				abs(previous_loss - loss)
			)
			previous_loss = loss

			recall_adjustments[i] = np.concatenate(
				adjustments, axis=1
			)

			norm_sum = np.linalg.norm(sum(recall_adjustments))
			sum_norm = sum([np.linalg.norm(p) for p in recall_adjustments])
			progress_factor = norm_sum / sum_norm
			
			if verbose:
				print 'loss:',loss,'\t','change:',change_in_loss
				print 'progress_factor:', progress_factor

			loss_converged = not(
				change_in_loss is None or change_in_loss > tolerance)
			progress_converged = progress_factor < self.progress_factor_threshold

			converged = loss_converged or progress_converged



	def save(self, path):
		np.savez(path, *self.get_param_values())

	def _load(self, path):
		loaded = np.load(path)
		self.params = []
		self.lengths = []
		for i in range(len(loaded.keys())):
			this_array = loaded['arr_%s' % i]
			self.params.append(shared(this_array))
			self.lengths.append(this_array.shape[1]-1)

	def load(self, path):
		self._load(path)
		self._build_and_compile_model()








