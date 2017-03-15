from parc_reader import ParcCorenlpReader as P
import matplotlib
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model, metrics
from sklearn import svm
from sklearn.svm import SVC, SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, make_scorer, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
import pickle
from harmonic_logistic import HarmonicLogistic
from harmonic_logistic_sklearn import HarmonicLogisticSK


def readData(datafile):
	df = pickle.load(open(datafile, 'rb'))

	#df = np.asarray(data)

	headers = list(df.columns.values)
	print len(headers)


	metaHeaders = headers[0:2]
	X_headers = headers[2:]

	numSourceFeats = len(filter(lambda s: s[0] == 's', X_headers)) 
	numCueFeats = len(filter(lambda s: s[0] == 'q', X_headers)) 
	numContentFeats = len(filter(lambda s: s[0] == 'c', X_headers)) 

	numHeaders = [numSourceFeats, numCueFeats, numContentFeats]
	print numHeaders


	metaData = df[metaHeaders]
	X_Values = df[X_headers].astype(np.float)

	metaData = metaData.as_matrix()
	X_Values = X_Values.as_matrix()

	X_train, X_test, train_metadata, test_metadata = train_test_split(X_Values, metaData, test_size=0.2, random_state=200)

	train_meta = train_metadata[:, 0]
	y_train = train_metadata[:, 1].astype(np.float) * 100

	test_meta = test_metadata[:, 0]
	y_test = test_metadata[:, 1].astype(np.float) * 100

	return X_train, X_test, y_train, y_test, train_meta, test_meta, X_headers, numHeaders

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

def bin_data(y_train, y_test):
	bins = np.array([10,20,30,40,50,60,70,80,90,100])

	binned_y_train = np.digitize(y_train, bins, right=False)
	binned_y_test = np.digitize(y_test, bins, right=False)

	return binned_y_train, binned_y_test

#####################   linear model   #####################

def linModel(X_train, y_train):
	linModel = linear_model.LinearRegression()
	#linModel.fit(X_train, y_train)
	return linModel

######################   Lasso Regularization    #####################	

def lasso(X_train, y_train):
	lasso = linear_model.LassoCV(cv = 5)
	#lasso.fit(X_train, y_train)
	return lasso

######################   Hyperparameter Tuning SVR    #####################	

def run_svr(X_train, y_train):

	tuned_parameters = [
		{'C': [1, 10, 100, 1000, 10000], 'kernel': ['linear']},
		{'C': [1, 10, 100, 1000, 10000], 'gamma': ['auto', 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']},
		{'C': [1, 10, 100, 1000, 10000], 'coef0': [0, 0.5, 1, 1.5, 2],'gamma': ['auto', 0.1, 0.01, 0.001, 0.0001], 'degree': [2,3,4,5,6], 'kernel': ['poly']},
		{'C': [1, 10, 100, 1000, 10000], 'gamma': ['auto', 0.1, 0.01, 0.001, 0.0001], 'coef0': [0, 0.5, 1, 1.5, 2],  'kernel': ['sigmoid']},
	]

	clf = GridSearchCV(SVR(C=1), tuned_parameters, cv=5, scoring='neg_mean_squared_error', verbose=True)
	clf.fit(X_train, y_train)
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

######################   Hyperparameter Tuning SVC    #####################	
def run_svc(X_train, y_train):

	tuned_parameters = [
		{'C': [1, 10, 100, 1000, 10000], 'kernel': ['linear']},
		{'C': [1, 10, 100, 1000, 10000], 'gamma': ['auto', 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']},
		{'C': [1, 10, 100, 1000, 10000], 'coef0': [0, 0.5, 1, 1.5, 2],'gamma': ['auto', 0.1, 0.01, 0.001, 0.0001], 'degree': [2,3,4,5,6], 'kernel': ['poly']},
		{'C': [1, 10, 100, 1000, 10000], 'gamma': ['auto', 0.1, 0.01, 0.001, 0.0001], 'coef0': [0, 0.5, 1, 1.5, 2],  'kernel': ['sigmoid']},
	]

	clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring='neg_mean_squared_error', verbose=True)
	clf.fit(X_train, y_train)
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
def myBinMeanSquaredError(real, predicted):
	bins = np.array([5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])

	realBin = np.digitize(real, bins, right=False)
	predBin = np.digitize(predicted, bins, right=False)
	return mean_squared_error(realBin, predBin)

def myBinMeanAbsoluteError(real, predicted):
	bins = np.array([5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])

	realBin = np.digitize(real, bins, right=False)
	predBin = np.digitize(predicted, bins, right=False)
	return mean_absolute_error(realBin, predBin)

def myBinAccuracy(real, predicted):
	bins = np.array([5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])

	realBin = np.digitize(real, bins, right=False)
	predBin = np.digitize(predicted, bins, right=False)

	return accuracy_score(realBin, predBin)


######################   CREATING BINNING SCORERS : per 10   #####################	
def myBinMeanSquaredErrorDecile(real, predicted):
	bins = np.array([10,20,30,40,50,60,70,80,90,100])

	realBin = np.digitize(real, bins, right=False)
	predBin = np.digitize(predicted, bins, right=False)
	return mean_squared_error(realBin, predBin)

def myBinMeanAbsoluteErrorDecile(real, predicted):
	bins = np.array([10,20,30,40,50,60,70,80,90,100])

	realBin = np.digitize(real, bins, right=False)
	predBin = np.digitize(predicted, bins, right=False)
	return mean_absolute_error(realBin, predBin)

def myBinAccuracyDecile(real, predicted):
	bins = np.array([10,20,30,40,50,60,70,80,90,100])

	realBin = np.digitize(real, bins, right=False)
	predBin = np.digitize(predicted, bins, right=False)

	return accuracy_score(realBin, predBin)


######################   QUANTITATIVE ERROR ANALYSIS    #####################	
def errorAnal(model,X_train, y_train):

	MSEscores = cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
	print MSEscores
	MSE = abs(np.mean(MSEscores))
	MAE = abs(np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')))

	print("Mean squared error: %.2f" % MSE)
	print("Root Mean squared error: %.2f" % math.sqrt(MSE))
	print("Mean Absolute Error: %.2f" % MAE)


	MSE_bins_scorer = make_scorer(myBinMeanSquaredError, greater_is_better=False)
	MSEbinned = abs(np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring=MSE_bins_scorer)))
	
	MAE_bins_scorer = make_scorer(myBinMeanAbsoluteError, greater_is_better=False)
	MAEbinned = abs(np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring=MAE_bins_scorer)))

	accuracy_bins_scorer = make_scorer(myBinAccuracy, greater_is_better=True)
	accuracy_binned = abs(np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring=accuracy_bins_scorer)))

	print("Binned (5): Mean squared error: %.2f" % MSEbinned)
	print("Binned (5): Root Mean squared error: %.2f" % math.sqrt(MSEbinned))
	print("Binned (5): Mean Absolute Error: %.2f" % MAEbinned)
	print("Binned (5): Accuracy: %.2f" % accuracy_binned)

	MSE_decile_bins_scorer = make_scorer(myBinMeanSquaredErrorDecile, greater_is_better=False)
	MSE_decile_binned = abs(np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring=MSE_decile_bins_scorer)))
	
	MAE_decile_bins_scorer = make_scorer(myBinMeanAbsoluteErrorDecile, greater_is_better=False)
	MAE_decile_binned = abs(np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring=MAE_decile_bins_scorer)))

	accuracy_decile_bins_scorer = make_scorer(myBinAccuracyDecile, greater_is_better=True)
	accuracy_decile_binned = abs(np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring=accuracy_decile_bins_scorer)))

	print("Binned (10): Mean squared error: %.2f" % MSE_decile_binned)
	print("Binned (10): Root Mean squared error: %.2f" % math.sqrt(MSE_decile_binned))
	print("Binned (10): Mean Absolute Error: %.2f" % MAE_decile_binned)
	print("Binned (10): Accuracy: %.2f" % accuracy_decile_binned)

	
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
	MAE = abs(np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')))

	print("Mean squared error: %.2f" % MSE)
	print("Root Mean squared error: %.2f" % math.sqrt(MSE))
	print("Mean Absolute Error: %.2f" % MAE)

	print
	print "classification report on training set"
	y_pred = cross_val_predict(model, X_train, y_train, cv=5)
	print(classification_report(y_train, y_pred))

	predictions = model.predict(X_train)
	for prediction in enumerate(predictions):
		index = prediction[0]
		print prediction[1], y_train[index], ((prediction[1] - y_train[index]) **2 )

	return predictions


def ablationTesting(model, X_train, y_train, headers):
	scores = []
	lastFeat = headers[0][0:3]
	currentFeat = headers[0][0:3]
	allFeats = [feature[0:3] for feature in headers]
	allFeatsSet = set([feature[0:3] for feature in headers])

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


	for indx, category in enumerate(categories):
		score = cross_val_score(model, X_train[:, category[0]:category[1]], y_train, scoring="r2", cv=5)
		scores.append((round(np.mean(score), 3), headers[indicesChange[indx]]))
	
	sortedScores = sorted(scores, reverse=True)
	score_numbers, feat_name = zip(*sortedScores)

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
	X_train, X_test, y_train, y_test, train_meta, test_meta, headers, headerNums = readData('data/verifiabilityNumFeatures_min5')
	print
	print
	
	print "----Linear Regression----"
	linRegression = linModel(X_train, y_train)
	print
	print "Error Analysis"
	print
	predictions = errorAnal(linRegression, X_train, y_train)
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
	predictions = errorAnal(lassoModel, X_train, y_train)
	print
	print "Quote Analysis"
	#top_bottom_quote_examples(predictions, y_train, test_meta)
	print
	print "Ablation Testing"
	ablationTesting(lassoModel,X_train, y_train, headers)
	print
	print
	
	'''
	print "----Edward's Harmonic Logisical Model----"
	print y_train
	hl_train = y_train / 100
	print hl_train
	harmonic_logistic, harmonic_logisticSK = harm_log(X_train, hl_train, headerNums)
	print
	print "Error Analysis"
	print hl_train
	harmonic_logistic.fit(X_train, hl_train, verbose=False)
	predictions = harmonic_logistic.predict(X_train)
	print
	print "NON SKLEARN PREDICTIONS"
	print "  Prediction  |  Real   |  Squared Error"
	for prediction in enumerate(predictions):
		index = prediction[0]
		if index == 10:
			break
		print prediction[1], hl_train[index], ((prediction[1] - hl_train[index]) **2 )

	print
	print "SKLEARN PREDICTIONS"
	predictions = errorAnal(harmonic_logisticSK, X_train, hl_train)
	print
	print "Quote Analysis"
	#top_bottom_quote_examples(predictions, y_train, test_meta)
	print
	#print "Ablation Testing"
	#ablationTesting(harmonic_logisticSK,X_train, y_train)
	#print
	print
	'''	
	
	

	print "----Support Vector Regression: Training + Hyper Parameter Tuning----"
	best_svr = run_svr(X_train, y_train)
	print
	print "Error Analysis"
	predictions = errorAnal(best_svr, X_train, y_train)
	print
	print "Quote Analysis"
	#top_bottom_quote_examples(predictions, y_test, test_meta)
	print
	print "Ablation Testing"
	ablationTesting(best_svr,X_train, y_train, headers)
	print
	print
	

	y_train_binned, y_test_binned = bin_data(y_train, y_test)

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


if __name__ == '__main__':
   main()

