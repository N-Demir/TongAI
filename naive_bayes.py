"""
Run Naive Bayes for a baseline

Results:
Overall KFold accuracy was 0.5049024870280908 (5 folds)

"""


import string
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, accuracy_score

import numpy as np

FILE_PATH = 'data/processed/final.csv'
DELIMITER = '||'

def readInData(file_path):
	# Let's dejsonify
	X = []
	Y = []

	with open(file_path, 'r') as f:
		for line in f:
			assert(len(line.split(DELIMITER)) == 4)

			X.append(line.split(DELIMITER)[2])
			Y.append(line.split(DELIMITER)[3])

	return (X, Y)

def trainKFoldModel(X, Y, model):
	accuracies = []
	recalls = []

	print('Starting KFold')

	kf = KFold(n_splits=5, shuffle=True)
	for train_index, test_index in kf.split(X):
		train_X, train_Y = X[train_index], Y[train_index]
		valid_X, valid_Y = X[test_index], Y[test_index]

		# Word features
		vectorizer = CountVectorizer()
		vectorizer.fit(train_X)
		train_X = vectorizer.transform(train_X)
		valid_X = vectorizer.transform(valid_X)

		# tfidf_transformator = TfidfTransformer()
		# tfidf_transformator.fit(train_X)
		# train_X = tfidf_transformator.transform(train_X)
		# valid_X = tfidf_transformator.transform(valid_X)

		print('Fitting the data')
		model.fit(train_X, train_Y)

		print('Scoring')
		predictions = model.predict(valid_X)

		accuracy = accuracy_score(valid_Y, predictions)
		recall = recall_score(valid_Y, predictions, average='weighted')
		print('Accuracy on one fold = {}'.format(accuracy))
		print('Recall on one fold = {}'.format(recall))

		accuracies.append(accuracy)
		recalls.append(recall)

		print('Completed a kfold')

	return (np.mean(accuracies), np.mean(recalls))

def main():

	X, Y = readInData(FILE_PATH)
	X = np.array(X)
	Y = np.array(Y)

	## Use comments to choose a particular model to train
	clf = MultinomialNB()
	# clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
	# clf = SGDClassifier(random_state=314159)

	## KFold training
	kfold_accuracy, kfold_recall = trainKFoldModel(X, Y, clf)
	print('Overall KFold accuracy was {} and recall was {}'.format(kfold_accuracy, kfold_recall))

if __name__ == "__main__":
	main()
