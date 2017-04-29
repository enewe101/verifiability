from parc_reader import ParcCorenlpReader as P
import math
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np
from tempfile import TemporaryFile
import pandas as pd
from sklearn import linear_model, metrics
from sklearn import svm
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, make_scorer, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
import pickle
from harmonic_logistic import HarmonicLogistic
from harmonic_logistic_sklearn import HarmonicLogisticSK
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
import copy
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

def readTrain(datafile):
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

	return X_train, y_train, train_meta,  X_headers, numHeaders, cutoff_values

def readTest(datafile):
	df = pickle.load(open(datafile, 'rb'))
	headers = list(df.columns.values)


	metaHeaders = headers[0:2]
	X_headers = headers[2:]

	sourcecolumns = list(filter(lambda s: s.startswith('s_8') and s != 's_8_sourceLength', X_headers))
	cuecolumns = list(filter(lambda s: s.startswith('q_2'), X_headers))
	dropColumns = list(filter(lambda s: s.startswith('c_3') or s.startswith('s_10'), X_headers))

	df.drop(sourcecolumns, axis=1, inplace=True)
	df.drop(dropColumns, axis=1, inplace=True)

	headers = list(df.columns.values)

	X_headers = headers[2:]


	print len(X_headers)
	print X_headers

	metaData = df[metaHeaders]
	X_Values = df[X_headers].astype(np.float)

	metaData = metaData.as_matrix()
	X_Values = X_Values.as_matrix()

	test_meta = metaData[:, 0]
	y_test = metaData[:, 1].astype(np.float) * 100
	X_test = X_Values
	
	return X_test, y_test, test_meta, headers



def getAttr(attr):
	filename = attr[0:8]
	folder = attr[4:6]
	top_folder = ''

	if int(folder) < 23: 
		top_folder = 'train/'
	elif int(folder) == 23:
		top_folder = 'test/'
	else:
		top_folder = 'dev/'


	parcFilePath = "/home/ndg/dataset/parc3/" + top_folder +  folder + '/' + filename + '.xml'
	corenlpPath =  "/home/ndg/dataset/ptb2-corenlp/CoreNLP/" + top_folder + filename + '.xml'
	raw_text = "/home/ndg/dataset/ptb2-corenlp/masked_raw/" + top_folder + '/' + filename

	parc_xml = open(parcFilePath).read()
	corenlp_xml = open(corenlpPath).read()
	raw_text = open(raw_text).read()

	article = P(corenlp_xml, parc_xml, raw_text)
	attribution = article.attributions[attr]
	return attribution

def bin_data(y_train, y_test, quintiles):
	bins = np.array(quintiles)

	binned_y_train = np.digitize(y_train, bins, right=True)
	binned_y_test = np.digitize(y_test, bins, right=True)

	return binned_y_train, binned_y_test

#####################   linear model   #####################

def linModel(X_train, y_train):
	linModel = linear_model.LinearRegression()
	#linModel.fit(X_train, y_train)
	return linModel

######################   Lasso Regularization    #####################	

def lasso(X_train, y_train):
	lasso = linear_model.Lasso()
	#lasso.fit(X_train, y_train)
	return lasso

######################   Hyperparameter Tuning SVR    #####################	

def run_svr(X_train, y_train):
	return SVR(kernel= 'rbf', C= 1000, gamma = 0.001)


	C_range = [0.01, 0.1, 1, 10]
	print len(C_range)
	gamma_range = np.logspace(-6, 3, 5)
	tuned_parameters = [
		{'C': C_range, 'kernel': ['linear']},
		{'C': [0.01, 0.1, 1, 10, 1000, 10000], 'gamma': [0.00001, .0001, .001, .1, 1], 'kernel': ['rbf']},
		#{'C': [0.01, 0.1, 1, 10] ,'gamma': [0.00001,.0001, .001, .1, 1], 'degree': [1,2,3,4,5], 'kernel': ['poly']},
		{'C': C_range, 'gamma': [0.00001,.0001, .001, .1, 1], 'kernel': ['sigmoid']},
	]

	clf = GridSearchCV(SVR(C=1), tuned_parameters, cv=5, scoring='neg_mean_squared_error', verbose=100)
	clf.fit(X_train, y_train)
	print clf.best_params_
	
	print("Grid scores on development set:")
	print
	means = clf.cv_results_['mean_test_score']
	stds = clf.cv_results_['std_test_score']
	for mean, std, params in zip(means, stds, clf.cv_results_['params']):
		print("%0.3f (+/-%0.03f) for %r"
			% (mean, std * 2, params))
	print
	

	return clf.best_estimator_

######################   Hyperparameter Tuning SVC    #####################	
def run_svc(X_train, y_train):
	return SVC(kernel= 'rbf', C= 10, gamma = 0.01).fit(X_train, y_train)
	

	#return SVC(kernel = 'linear', C = 0.1).fit(X_train, y_train)

	C_range = [0.01, 0.1, 1, 10, 100, 1000]
	print len(C_range)
	gamma_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
	tuned_parameters = [
		{'C': C_range, 'kernel': ['linear']},
		{'C': C_range, 'gamma': gamma_range, 'kernel': ['rbf']},
		#{'C': [0.01, 0.1, 1, 10] ,'gamma': [.0001, .001, .1, 1], 'degree': [1, 2,3,4,5], 'kernel': ['poly']},
		{'C': C_range, 'gamma': gamma_range, 'kernel': ['sigmoid']},
	]

	cv = StratifiedShuffleSplit(n_splits=5, random_state=42)
	clf = GridSearchCV(SVC(), param_grid=tuned_parameters, cv=cv, verbose=100)

	clf.fit(X_train, y_train)
	print clf.best_params_


	print(clf.best_params_)
	
	print("Grid scores on development set:")
	print
	means = clf.cv_results_['mean_test_score']
	stds = clf.cv_results_['std_test_score']
	for mean, std, params in zip(means, stds, clf.cv_results_['params']):
		print("%0.3f (+/-%0.03f) for %r"
			% (mean, std * 2, params))
	print
	

	return clf.best_estimator_

############## Edward's Harmonic Logistic ###############
def harm_log(X_train, y_train, headersNums):
	harmonic_logistic = HarmonicLogistic(lengths=headersNums)
	#harmonic_logistic.fit(X_train, y_train, verbose=False)

	harmonic_logisticSK = HarmonicLogisticSK(lengths=headersNums)
	#harmonic_logisticSK.fit(X_train, y_train)

	return harmonic_logistic, harmonic_logisticSK


######################   CREATING BINNING SCORERS : per 5   #####################	
def myBinMeanSquaredError(real, predicted, **kwargs):
	bins_quintiles = list(kwargs.iteritems())[0][1]
	bins = np.array(bins_quintiles)
	print bins

	realBin = np.digitize(real, bins, right=True)
	predBin = np.digitize(predicted, bins, right=True)
	return mean_squared_error(realBin, predBin)

def myBinMeanAbsoluteError(real, predicted, **kwargs):
	bins_quintiles = list(kwargs.iteritems())[0][1]
	bins = np.array(bins_quintiles)

	realBin = np.digitize(real, bins, right=True)
	predBin = np.digitize(predicted, bins, right=True)
	return mean_absolute_error(realBin, predBin)

def myBinAccuracy(real, predicted, **kwargs):
	bins_quintiles = list(kwargs.iteritems())[0][1]
	bins = np.array(bins_quintiles)

	realBin = np.digitize(real, bins, right=True)
	predBin = np.digitize(predicted, bins, right=True)

	return accuracy_score(realBin, predBin)

def myBinAccuracyQuintiles(real, predicted, quintiles):
	bins = np.array(quintiles)

	realBin = np.digitize(real, bins, right=True)
	predBin = np.digitize(predicted, bins, right=True)

	return accuracy_score(realBin, predBin)



######################   QUANTITATIVE ERROR ANALYSIS    #####################	
def errorAnal(model,X_train, y_train, quintiles):

	MSEscores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
	print MSEscores
	MSE = abs(np.mean(MSEscores))
	MAE = abs(np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')))


	print("Mean squared error: %.5f" % MSE)
	print("Root Mean squared error: %.5f" % math.sqrt(MSE))
	print("Mean Absolute Error: %.5f" % MAE)


	MSE_bins_scorer = make_scorer(myBinMeanSquaredError, greater_is_better=False, bins = quintiles)
	MSEbinned = abs(np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring=MSE_bins_scorer)))
	
	MAE_bins_scorer = make_scorer(myBinMeanAbsoluteError, greater_is_better=False, bins = quintiles)
	MAEbinned = abs(np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring=MAE_bins_scorer)))

	accuracy_bins_scorer = make_scorer(myBinAccuracy, greater_is_better=True, bins = quintiles)
	accuracy_binned = abs(np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring=accuracy_bins_scorer)))

	print("Binned (5): Mean squared error: %.5f" % MSEbinned)
	print("Binned (5): Root Mean squared error: %.5f" % math.sqrt(MSEbinned))
	print("Binned (5): Mean Absolute Error: %.5f" % MAEbinned)
	print("Binned (5): Accuracy: %.5f" % accuracy_binned)
	
	model.fit(X_train, y_train)
	predictions = model.predict(X_train)
	print "  Prediction  |  Real   |  Squared Error"
	for prediction in enumerate(predictions):
		index = prediction[0]
		if index == 30:
			break
		print prediction[1], y_train[index], ((prediction[1] - y_train[index]) **2 )
	

	return predictions

def binnedErrorAnalysis(model,X_train, y_train):
	MSE = abs(np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')))
	print MSE

	MAE = abs(np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')))
	print MAE
	accuracy = abs(np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')))
	print accuracy
	print cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')


	print("Mean squared error: %.5f" % MSE)
	print("Root Mean squared error: %.5f" % math.sqrt(MSE))
	print("Mean Absolute Error: %.5f" % MAE)
	print("Accuracy: %.5f" % accuracy)

	print
	print "classification report on training set"
	y_pred = cross_val_predict(model, X_train, y_train, cv=5)
	print(classification_report(y_train, y_pred))
	print confusion_matrix(y_train, y_pred)

	predictions = model.predict(X_train)
	print 
	for prediction in enumerate(predictions):
		index = prediction[0]
		if index == 10:
			break
		print prediction[1], y_train[index], ((prediction[1] - y_train[index]) **2 )

	return predictions

def chunkList(lst, n):
	return [ lst[i::n] for i in xrange(n) ]

def hl_errorAnal(harmonic_logistic, X_train, hl_train, headerNums, quintiles):
	#harmonic_logistic.fit(X_train, hl_train)
	#predictions = harmonic_logistic.predict(X_train)
	#print predictions[0:10]
	

	#X_train, X1, hl_train, y1 = train_test_split(X_train, hl_train, test_size=0.9, random_state=200)


	X, X1, y, y1 = train_test_split(X_train, hl_train, test_size=0.2, random_state=200)
	X, X2, y, y2 = train_test_split(X, y, test_size=0.25, random_state=200)
	X, X3, y, y3 = train_test_split(X, y, test_size=0.33, random_state=200)
	X5, X4, y5, y4 = train_test_split(X, y, test_size=0.5, random_state=200)

	split1 = [(np.concatenate([X2,X3,X4,X5]), X1), (np.concatenate([y2,y3,y4,y5]), y1)]
	split2 = [(np.concatenate([X1,X3,X4,X5]), X2), (np.concatenate([y1,y3,y4,y5]), y2)]
	split3 = [(np.concatenate([X1,X2,X4,X5]), X3), (np.concatenate([y1,y2,y4,y5]), y3)]
	split4 = [(np.concatenate([X1,X2,X3,X5]), X4), (np.concatenate([y1,y2,y3,y5]), y4)]
	split5 = [(np.concatenate([X1,X2,X3,X4]), X5), (np.concatenate([y1,y2,y3,y4]), y5)]

	splits = [split1, split2, split3, split4, split5]

	resultsAccuracy = []
	resultsMSE = []

	quintiles_bins = [x / 100 for x in quintiles]
	print quintiles_bins


	for split in splits:
		xtrain = split[0][0]
		xtest = split[0][1]

		ytrain = split[1][0]
		ytest = split[1][1]

		print len(xtrain), len(ytrain)
		print len(xtest), len(ytest)
		harmonic_logistic = HarmonicLogistic(lengths=headerNums)
		harmonic_logistic.fit(xtrain, ytrain, verbose=False)
		predictions = harmonic_logistic.predict(xtest)
		#print predictions[0:10]
		#print ytest[0:10]
		MSE = mean_squared_error(ytest, predictions)
		accuracy = myBinAccuracyQuintiles(ytest, predictions,  quintiles_bins)
		MAW = mean_absolute_error(ytest, predictions)

		print MSE
		print accuracy
		print MAW

		resultsAccuracy.append(accuracy)
		resultsMSE.append(MSE)

	print resultsAccuracy
	print resultsMSE
	print 'average accuracy: ' + str(np.mean(resultsAccuracy))
	print 'average MSE: ' + str(abs(np.mean(resultsMSE)))
	print 'average root MSE: ' + str(math.sqrt(abs(np.mean(resultsMSE))))

def final_test(model, X_train, y_train, X_test, y_test, quintiles):

	model.fit(X_train, y_train)
	predictions = model.predict(X_test)

	print "MSE ON FINAL TEST"
	print mean_squared_error(y_test, predictions)

	print "MAE ON FINAL TEST"
	print mean_absolute_error(y_test, predictions)

	print "ACCURACY BINS"
	print myBinAccuracy(y_test, predictions, bins = quintiles)




def ablationTesting(model, X_train, y_train, headers):
	scores = []


	sqcFeats = [feature[0:1] for feature in headers]
	sqcFeatsSet = set([feature[0:1] for feature in headers])
	sqcfeatNames = sorted(list(sqcFeatsSet))
	sqcIndicesChange = []

	for elem in sqcfeatNames:
		sqcIndicesChange.append(sqcFeats.index(elem))

	sqcIndicesChange = sorted(sqcIndicesChange)

	sqcCategories = []
	for index, realIndex in enumerate(sqcIndicesChange):
		if index == 0:
			sqcCategories.append((0,sqcIndicesChange[1]))
		elif index == len(sqcIndicesChange) - 1:
			sqcCategories.append((realIndex, len(headers)))
		else:
			sqcCategories.append((realIndex,sqcIndicesChange[index + 1]))

	sqcScoresWithout = []
	sqcScores = []
	
	for indx, category in enumerate(sqcCategories):
		score = cross_val_score(model, X_train[:, category[0]:category[1]], y_train, scoring="neg_mean_squared_error", cv=5)
		sqcScores.append((round(math.sqrt(abs(np.mean(score))), 3), headers[sqcIndicesChange[indx]]))

		xtrain2 = np.delete(X_train, np.s_[category[0]:category[1]], axis=1) 
		print len(xtrain2[0])

		scoreWO = cross_val_score(model, xtrain2, y_train, scoring="neg_mean_squared_error", cv=5)
		sqcScoresWithout.append((round(math.sqrt(abs(np.mean(scoreWO))), 3), headers[sqcIndicesChange[indx]]))
	
	sqcSortedScores = sorted(sqcScores, reverse=False)
	sqc_sortedScoresWithout = sorted(sqcScoresWithout, reverse=True)
	print "With each feature seperately: ablated on source/cue/content"
	print sqcSortedScores
	print "With only not that feature: ablated on source/cue/content"
	print sqc_sortedScoresWithout







	allFeats = [feature[0:4] for feature in headers]
	allFeatsSet = set([feature[0:4] for feature in headers])

	featNames = sorted(list(allFeatsSet))

	indicesChange = []
	 

	for elem in featNames:
		indicesChange.append(allFeats.index(elem))

	indicesChange = sorted(indicesChange)

	categories = []
	for index, realIndex in enumerate(indicesChange):
		if index == 0:
			categories.append((0,indicesChange[1]))
		elif index == len(indicesChange) - 1:
			categories.append((realIndex, len(headers)))
		else:
			categories.append((realIndex,indicesChange[index + 1]))

	scoresWithout = []

	for indx, category in enumerate(categories):
		score = cross_val_score(model, X_train[:, category[0]:category[1]], y_train, scoring="neg_mean_squared_error", cv=5)
		scores.append((round(math.sqrt(abs(np.mean(score))), 3), headers[indicesChange[indx]]))

		xtrain2 = np.delete(X_train, np.s_[category[0]:category[1]], axis=1) 
		print len(xtrain2[0])

		scoreWO = cross_val_score(model, xtrain2, y_train, scoring="neg_mean_squared_error", cv=5)
		scoresWithout.append((round(math.sqrt(abs(np.mean(scoreWO))), 3), headers[indicesChange[indx]]))
	
	sortedScores = sorted(scores, reverse=False)
	score_numbers, feat_name = zip(*sortedScores)

	sortedScoresWithout = sorted(scoresWithout, reverse=True)
	print "With each feature seperately"
	print sortedScores
	print "With only not that feature"
	print sortedScoresWithout

def prettyprintquote(attr):
	attrid = attr['id']
	tokens = attr['content'] + attr['source'] + attr['cue']

	tokens = sorted(tokens, key=lambda x: x['character_offset_begin'], reverse=False)
	string = attrid + ': ' + ' '.join(token['word'] for token in tokens)

	return string

def top_bottom_quote_examples(predictions, y_test, metadata):
	differences = abs(predictions - y_test)

	#min
	min_index = np.argmin(differences)
	minAttr = metadata[min_index]
	minQuote = getAttr(minAttr)
	minString = prettyprintquote(minQuote)

	#max
	max_index = np.argmax(differences)
	maxAttr = metadata[max_index]
	maxQuote = getAttr(maxAttr)
	maxString = prettyprintquote(maxQuote)

	#median
	medianIndex = np.argsort(differences)[len(differences)//2]
	medAttr = metadata[medianIndex]
	medQuote = getAttr(medAttr)
	medString = prettyprintquote(medQuote)

	print "Very Correct Attribution, difference: " + str(differences[min_index])
	print minString

	print

	print "Very Incorrect Attribution, difference: " + str(differences[max_index])
	print maxString
	print

	print "Middle of the Way Attribution, difference: " + str(differences[medianIndex])
	print medString
	print





def main():

	print "Reading Data"
	X_train, y_train, train_meta, headers, headerNums, quintiles = readTrain('data/verifiabilityNumFeatures_len_5_liwc_train_20comp')
	X_test, y_test, test_meta, headers2 = readTest('data/verifiabilityNumFeatures_len_5_liwc_test_20comp')
	print X_train


	print list(set(headers) - set(headers2))

	print len(X_train[0])
	print len(X_test[0])

	all_X = np.concatenate((X_train, X_test), axis=0)
	all_y = np.concatenate((y_train, y_test), axis=0)


	undertwenty = [x for x in y_train if x <= 20]
	underforty = [x for x in y_train if x > 20 and x <= 40 ]
	undersizty = [x for x in y_train if x > 40 and x <= 60 ]
	undereighty = [x for x in y_train if x > 60 and x <= 80 ]
	underhundred = [x for x in y_train if x > 80 and x <= 100 ]



	print len(undertwenty)
	print len(underforty)
	print len(undersizty)
	print len(undereighty)
	print len(underhundred)
	'''
	np.savez('data/train.npz', x = X_train, y = y_train)
	npzfile = np.load('data/train.npz')
	print npzfile
	print npzfile['x']
	print
	entity.ent
	'''
	#try NN?
	
	print "----Linear Regression----"
	linRegression = linModel(X_train, y_train)
	print
	print "Error Analysis"
	print
	predictions = errorAnal(linRegression, X_train, y_train, quintiles)
	print
	print "Quote Analysis"
	#top_bottom_quote_examples(predictions, y_train, test_meta)
	print
	print "Ablation Testing"
	print
	ablationTesting(linRegression,X_train, y_train, headers)
	print
	print
	
	print "----Linear Regression with Lasso Regularization----"
	lassoModel = lasso(X_train, y_train)
	print
	print "Error Analysis"
	predictions = errorAnal(lassoModel, X_train, y_train, quintiles)
	print
	print "Quote Analysis"
	#top_bottom_quote_examples(predictions, y_train, test_meta)
	print
	print "Ablation Testing"
	ablationTesting(lassoModel,X_train, y_train, headers)
	print
	print
	
	
	
	print "----Edward's Harmonic Logisical Model----"

	print y_train
	hl_train = y_train / 100
	print hl_train
	harmonic_logistic, harmonic_logisticSK = harm_log(X_train, hl_train, headerNums)
	print
	print "Error Analysis"
	print hl_train

	predictions = hl_errorAnal(harmonic_logistic, X_train, hl_train, headerNums, quintiles)
	
	
	
	
	
	print "----Support Vector Regression: Training + Hyper Parameter Tuning----"
	best_svr = run_svr(X_train, y_train)
	best_svr.fit(X_train, y_train)

	print
	print "Error Analysis"
	predictions = errorAnal(best_svr, X_train, y_train, quintiles)
	print
	print "Quote Analysis"
	#top_bottom_quote_examples(predictions, y_test, test_meta)
	print
	print "Ablation Testing"
	ablationTesting(best_svr,X_train, y_train, headers)
	print
	print
	
	final_test(best_svr, X_train, y_train, X_test, y_test, quintiles)

	print 'retraining on all data'

	new_svr = run_svr(all_X, all_y)
	new_svr.fit(all_X, all_y)

	errorAnal(new_svr, all_X, all_y, quintiles)
	joblib.dump(new_svr, 'best_ver_regressionSVR(20).pkl')

	
	
	
	y_train_binned, y_test_binned = bin_data(y_train, y_test, quintiles)

	print "----Multiclass Support Vector Machine: Training + Hyper Parameter Tuning----"
	best_svc = run_svc(X_train, y_train_binned)
	print
	print "Error Analysis"
	predictions = binnedErrorAnalysis(best_svc, X_train, y_train_binned)
	print
	print "Quote Analysis"
	#top_bottom_quote_examples(predictions, y_test_binned, test_meta)
	print
	print "Ablation Testing"
	ablationTesting(best_svc,X_train, y_train_binned, headers)
	print
	print
	

	y_train_binned, y_test_binned = bin_data(y_train, y_test, quintiles)

	



if __name__ == '__main__':
   main()

