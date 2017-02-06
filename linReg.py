import matplotlib

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, metrics

from sklearn.svm import SVR
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC



from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
matplotlib.use('Agg')
import pickle

data = pickle.load(open('verifiabilityNumFeatures', 'rb'))

df = np.asarray(data)
df = df.astype(np.float)

Y_values = df[:, 0]
X_Values = df[:,1:]

X_train, X_test, y_train, y_test = train_test_split(X_Values, Y_values, test_size=0.33)

#####################   linear model   #####################

print "LINEAR MODEL"
linModel = linear_model.LinearRegression()
linModel.fit(X_train, y_train)

# The coefficients
#print('Coefficients: \n', linModel.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((linModel.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % linModel.score(X_test, y_test))

scores = cross_val_score(linModel, X_train, y_train, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

predictions = linModel.predict(X_test)
for prediction in enumerate(predictions):
	index = prediction[0]
	print prediction[1], y_test[index]

######################   Lasso Regularization    #####################
print 
print "LASSO REGULARIZATION LINEAR MODEL"
lasso = linear_model.LassoCV(cv = 5)
lasso.fit(X_train, y_train)
print("Mean squared error: %.2f"
      % np.mean((lasso.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % lasso.score(X_test, y_test))

scores = cross_val_score(lasso, X_train, y_train, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


predictions = lasso.predict(X_test)
for prediction in enumerate(predictions):
	index = prediction[0]
	print prediction[1], y_test[index]

