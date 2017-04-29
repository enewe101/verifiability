import random
import copy
import numpy as np
import pickle
import os
import get_features
from SETTINGS import DATA_DIR
import harmonic_logistic
from sklearn.linear_model import LinearRegression

TRAINING_DATA_PATH = os.path.join(
	DATA_DIR, 'features', 'verifiabilityNumFeatures_len_5_liwc_train_20comp')

NUM_FOLDS = 5
def train_validate_harmonic_logistic(model_type='radial'):
	"""
	Splits training data into a training and validation set, trains the model,
	and then tests it on the validation set
	"""

	# Get the training data
	X_train, y_train, numHeaders, cutoff_values = readTrain()

	# verifiabilities are as percentages.  Convert to fraction
	y_train = y_train / 100.

	# Which model will we be training?
	if model_type == 'radial':
		model_class = harmonic_logistic.HarmonicRadialLogistic
	elif model_type == 'linear':
		model_class = harmonic_logistic.HarmonicLogistic
	elif model_type == 'logistic':
		model_class = harmonic_logistic.Logistic
	else:
		raise ValueError('unexpected model type: %s' % model_type)

	mean_squared_errors = []
	for fold in range(NUM_FOLDS):

		X_fold_train, X_fold_test = get_fold(X_train, NUM_FOLDS, fold)
		y_fold_train, y_fold_test = get_fold(y_train, NUM_FOLDS, fold)

		#regressor = model_class(
		#	numHeaders,
		#	learning_rate=1.,
		#	clip = 0.0001,
		#	progress_tolerance=0.01, 
		#	loss_tolerance=1e-20
		#)
		#regressor.fit(X_fold_train, y_fold_train, verbose=True)

		regressor = LinearRegression()
		regressor.fit(X_fold_train, y_fold_train)
		predicted_y = regressor.predict(X_fold_test)

		mean_squared_errors.append(
			np.mean((predicted_y - y_fold_test)**2))

	rms = np.sqrt(np.mean(mean_squared_errors))
	print 'RMS:', rms


def get_fold(data, num_folds, this_fold):
	if this_fold < 0 or this_fold >= num_folds:
		raise ValueError('``this_fold`` must be in [0,num_folds-1]')

	# Reproducibly create a random permutation
	random.seed(0)
	permutation = range(len(data))
	random.shuffle(permutation)

	# Get the slice of the random permutation that will correspond to the test
	# part of the fold
	start = int(this_fold * float(len(data)) / num_folds)
	end = int((this_fold+1) * float(len(data)) / num_folds)
	fold_test_indices = permutation[start:end]

	# Everything else is the training part of the fold
	fold_train_indices = permutation[:start] + permutation[end:]

	fold_test = np.array([data[i] for i in fold_test_indices])
	fold_train = np.array([data[i] for i in fold_train_indices])
	return fold_train, fold_test






def readTrain(datafile=TRAINING_DATA_PATH):
	df = pickle.load(open(datafile, 'rb'))
	df = df.rename(index=str, columns={"s_8_contentLength": "c_2_contentLength"})

	headers = list(df.columns.values)


	metaHeaders = headers[0:2]
	X_headers = headers[2:]

	sourcecolumns = list(filter(lambda s: s.startswith('s_8') and s != 's_8_sourceLength', X_headers))
	cuecolumns = list(filter(lambda s: s.startswith('q_2'), X_headers))
	dropColumns = list(filter(lambda s: s.startswith('c_3') or s.startswith('s_10'), X_headers))


	df.drop(sourcecolumns, axis=1, inplace=True)
	df.drop(dropColumns, axis=1, inplace=True)

	#df.drop(cuecolumns, axis=1, inplace=True)
	headers = list(df.columns.values)

	X_headers = headers[2:]


	#df = np.asarray(data)

	print len(X_headers)
	print X_headers

	numSourceFeats = len(filter(lambda s: s[0] == 's', X_headers)) 
	numCueFeats = len(filter(lambda s: s[0] == 'q', X_headers)) 
	numContentFeats = len(filter(lambda s: s[0] == 'c', X_headers)) 
	#print filter(lambda s: s[0] == 's', X_headers)

	numHeaders = [numSourceFeats, numCueFeats, numContentFeats]
	print numHeaders


	metaData = df[metaHeaders]
	X_Values = df[X_headers].astype(np.float)

	metaData = metaData.as_matrix()
	X_Values = X_Values.as_matrix()

	#X_train, X_test, train_metadata, test_metadata = train_test_split(X_Values, metaData, test_size=0.2, random_state=200)

	#train_meta = train_metadata[:, 0]
	#y_train = train_metadata[:, 1].astype(np.float) * 100

	#test_meta = test_metadata[:, 0]
	#y_test = test_metadata[:, 1].astype(np.float) * 100

	train_meta = metaData[:, 0]
	y_train = metaData[:, 1].astype(np.float) * 100
	X_train = X_Values


	y_copy = copy.deepcopy(y_train).tolist()
	sorted_ycopy = sorted(y_copy)
	cutoff_values = []
	length = len(sorted_ycopy)
	fifths = length/5

	cutoff_values = [sorted_ycopy[int(fifths)], sorted_ycopy[int(fifths) * 2], sorted_ycopy[int(fifths) * 3], sorted_ycopy[int(fifths) * 4], sorted_ycopy[length-1]]
	print cutoff_values

	#return X_train, y_train, train_meta,  X_headers, numHeaders, cutoff_values
	return X_train, y_train, numHeaders, cutoff_values
