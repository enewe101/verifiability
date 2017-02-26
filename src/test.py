import harmonic_logistic as hl
import numpy as np
from theano import function, shared, tensor as T
from unittest import TestCase, main


class TestLogistic(TestCase):

	def test_basic_calculations(self):
		input_matrix, params, output_vector = hl.logistic(1)
		f = function([input_matrix], output_vector)
		params.set_value(np.array([[2]], dtype='float64'))

		output = f([[-1], [0], [1]])
		expected = np.array(
			[[1/(1+np.exp(2))], [1/(1+np.exp(0))], [1/(1+np.exp(-2))]]
		)
		self.assertTrue(np.array_equal(output, expected))

	def test_more_calculations(self):
		input_matrix, params, output_vector = hl.logistic(4)
		f = function([input_matrix], output_vector)
		params.set_value(np.array([[-2,-1,0,1]], dtype='float64'))

		output = f([[-1,0,1,2], [0,1,2,3], [1,2,3,4]])
		expected = np.array([
			[1/(1+np.exp(-(2+0+0+2)))],
			[1/(1+np.exp(-(0-1+0+3)))],
			[1/(1+np.exp(-(-2-2+0+4)))]
		])
		self.assertTrue(np.array_equal(output, expected))

	def test_supplying_inputs(self):
		input_matrix = T.dmatrix()
		init_params = shared(np.array([[2]], dtype='float64'))
		_, _, output_vector = hl.logistic(1, input_matrix, init_params)
		f = function([input_matrix], output_vector)

		output = f([[-1], [0], [1]])
		expected = np.array(
			[[1/(1+np.exp(2))], [1/(1+np.exp(0))], [1/(1+np.exp(-2))]]
		)
	
		self.assertTrue(np.array_equal(output, expected))


	def test_supplying_inputs_more(self):
		input_matrix = T.dmatrix()
		init_params = shared(np.array([[-2,-1,0,1]], dtype='float64'))
		_, _, output_vector = hl.logistic(4, input_matrix, init_params)
		f = function([input_matrix], output_vector)

		output = f([[-1,0,1,2], [0,1,2,3], [1,2,3,4]])
		expected = np.array([
			[1/(1+np.exp(-(2+0+0+2)))],
			[1/(1+np.exp(-(0-1+0+3)))],
			[1/(1+np.exp(-(-2-2+0+4)))]
		])
		self.assertTrue(np.array_equal(output, expected))




class TestHarmonicLogisticComputationGraph(TestCase):


	def test_basic_calculations(self):
		(input_matrix, logistic_weights, logistic_outputs, output_vector
			) = hl.harmonic_logistic([1])

		f = function([input_matrix], output_vector)
		logistic_weights[0].set_value(np.array([[2]], dtype='float64'))

		output = f([[-1], [0], [1]])
		expected = np.array(
			[1/(1+np.exp(2)), 1/(1+np.exp(0)), 1/(1+np.exp(-2))]
		)
		self.assertTrue(np.array_equal(output, expected))


	def test_more_calculations(self):
		(input_matrix, logistic_weights, logistic_outputs, output_vector
			) = hl.harmonic_logistic([4,3])

		f = function([input_matrix], output_vector)
		logistic_weights[0].set_value(np.array([[-1,0,1,2]], dtype='float64'))
		logistic_weights[1].set_value(np.array([[-1,0,1]], dtype='float64'))

		output = f([
			[-1,0,1,2, 3,4,5], 
			[0,1,2,3, 4,5,6], 
			[1,2,3,4, 5,6,7]
		])
		expected = np.array([
			1/((1+np.exp(-(1 + 0 + 1 + 4)) + (1+np.exp(-(-3 + 0 + 5))))),
			1/((1+np.exp(-(0 + 0 + 2 + 6)) + (1+np.exp(-(-4 + 0 + 6))))),
			1/((1+np.exp(-(-1 + 0 + 3 + 8)) + (1+np.exp(-(-5 + 0 + 7)))))
		])
		self.assertTrue(np.array_equal(output, expected))


	def test_supplying_inputs(self):
		input_matrix = T.dmatrix()
		params = [shared(np.array([[2]], dtype='float64'))]
		_, _, logistic_outputs, output_vector = hl.harmonic_logistic(
			[1], input_matrix, params)

		f = function([input_matrix], output_vector)

		output = f([[-1], [0], [1]])
		expected = np.array(
			[1/(1+np.exp(2)), 1/(1+np.exp(0)), 1/(1+np.exp(-2))]
		)
		self.assertTrue(np.array_equal(output, expected))


	def test_supplying_inputs_more(self):
		input_matrix = T.dmatrix()
		params = [
			shared(np.array([[-1,0,1,2]], dtype='float64')),
			shared(np.array([[-1,0,1]], dtype='float64'))
		]
		_, _, logistic_outputs, output_vector = hl.harmonic_logistic(
			[4,3], input_matrix, params)

		f = function([input_matrix], output_vector)

		output = f([
			[-1,0,1,2, 3,4,5], 
			[0,1,2,3, 4,5,6], 
			[1,2,3,4, 5,6,7]
		])
		expected = np.array([
			1/((1+np.exp(-(1 + 0 + 1 + 4)) + (1+np.exp(-(-3 + 0 + 5))))),
			1/((1+np.exp(-(0 + 0 + 2 + 6)) + (1+np.exp(-(-4 + 0 + 6))))),
			1/((1+np.exp(-(-1 + 0 + 3 + 8)) + (1+np.exp(-(-5 + 0 + 7)))))
		])
		self.assertTrue(np.array_equal(output, expected))


class TestHarmonicLogisticRegressor(TestCase):

	def test_loss_function(self):
		regressor = hl.HarmonicLogistic((4,3))
		regressor.params[0].set_value(np.array([[-1,0,1,2]], dtype='float64'))
		regressor.params[1].set_value(np.array([[-1,0,1]], dtype='float64'))

		expected_outputs = np.array([
			1/((1+np.exp(-(1 + 0 + 1 + 4)) + (1+np.exp(-(-3 + 0 + 5))))),
			1/((1+np.exp(-(0 + 0 + 2 + 6)) + (1+np.exp(-(-4 + 0 + 6))))),
			1/((1+np.exp(-(-1 + 0 + 3 + 8)) + (1+np.exp(-(-5 + 0 + 7)))))
		])

		# Define the target to be offset elementwise by 1 from what the unit
		# will calculate
		offset = 1.0
		target = expected_outputs + offset

		# The expected loss is the sum of squares of offsets
		expected_loss = offset**2 * len(target)

		# We'll first test the output when running in prediction mode
		outputs = regressor.predict([
			[-1,0,1,2, 3,4,5], 
			[0,1,2,3, 4,5,6], 
			[1,2,3,4, 5,6,7]
		])

		# Did we get the expected output from prediction mode?
		self.assertTrue(np.array_equal(outputs, expected_outputs))

		# Next we'll test the output when running in training mode
		outputs, loss = regressor.train([
			[-1,0,1,2, 3,4,5], 
			[0,1,2,3, 4,5,6], 
			[1,2,3,4, 5,6,7]
		], target)

		# Did we get the expected output and loss from prediction mode?
		self.assertTrue(np.array_equal(outputs, expected_outputs))
		self.assertTrue(np.array_equal(loss, expected_loss))

		# Having run one iteration in prediction mode, the loss should be
		# less when we run another
		_, new_loss = regressor.train([
			[-1,0,1,2, 3,4,5], 
			[0,1,2,3, 4,5,6], 
			[1,2,3,4, 5,6,7]
		], target)
		self.assertTrue(new_loss < loss)



	def test_fit(self):

		# Build a "hidden model" that will generate the data.  We will then
		# train a model on the generated data, and hopefully the trained model
		# will look like the hidden model.
		target_regressor = hl.HarmonicLogistic((4,3))
		target_regressor.params[0].set_value(
			np.array([[-1,0,1,2]], dtype='float64'))
		target_regressor.params[1].set_value(
			np.array([[-1,0,1]], dtype='float64'))

		# Generate data by collecting outputs of hidden model on random inputs
		inputs = np.random.rand(100, 4+3)
		targets = target_regressor.predict(inputs)

		# Here is the model that will get trained
		regressor_to_train = hl.HarmonicLogistic((4,3), learning_rate=1)

		# Now train the model
		regressor_to_train.fit(inputs, targets, tolerance=1e-20, verbose=False)

		self.assertTrue(
			np.isclose(
				target_regressor.params[0].get_value(),
				regressor_to_train.params[0].get_value()
			).all()
		)

		self.assertTrue(
			np.isclose(
				target_regressor.params[1].get_value(),
				regressor_to_train.params[1].get_value()
			).all()
		)


if __name__ == '__main__':
	main()
