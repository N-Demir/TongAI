"""
Run Naive Bayes for a baseline

Results:
Overall KFold accuracy was 0.5049024870280908 (5 folds)

On just brain things: 0.7617668514586854 (5 folds)

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
import matplotlib.pyplot as plt

import numpy as np

FILE_PATH = 'data/processed/final.csv'
DELIMITER = '|'
CLASSES = ['mets', 'memory', 'multiple sclerosis', 'epilepsy', 'stereo/cyberknife', 'routine brain', 'sella', 'tumor brain', 'complex headache', 'brain trauma', 'stroke']

KFOLDS = 5

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

def class_accuracy(preds, y):
    class_correct = np.zeros(len(CLASSES))
    class_counts = np.zeros(len(CLASSES))
    for idx, pred in enumerate(preds):
        pred = int(pred)
        if pred == y[idx]:
            class_correct[pred] += 1
        class_counts[pred] += 1

    return class_correct, class_counts

def trainKFoldModel(X, Y, model):
	accuracies = []
	recalls = []
	class_correct = np.zeros(len(CLASSES))
	class_counts = np.zeros(len(CLASSES))

	print('Starting KFold')

	kf = KFold(n_splits=KFOLDS, shuffle=True)
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

		fold_class_correct, fold_class_counts = class_accuracy(predictions, valid_Y)
		class_correct += fold_class_correct
		class_counts += fold_class_counts

		accuracies.append(accuracy)
		recalls.append(recall)

		print('Completed a kfold')

	return np.mean(accuracies), np.mean(recalls), class_correct / KFOLDS, class_counts / KFOLDS 

def main():

	X, Y = readInData(FILE_PATH)
	X = np.array(X)
	Y = np.array(Y)

	## Use comments to choose a particular model to train
	clf = MultinomialNB()
	# clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
	# clf = SGDClassifier(random_state=314159)

	## KFold training
	kfold_accuracy, kfold_recall, class_correct, class_counts = trainKFoldModel(X, Y, clf)
	print('Overall KFold accuracy was {} and recall was {}'.format(kfold_accuracy, kfold_recall))

	class_accuracy = np.divide(class_correct, class_counts)
	print('Val class accuracies: {}'.format([ CLASSES[idx] + ' ' + "{:.3f}".format(accuracy) for idx, accuracy in enumerate(class_accuracy) ] ))
	print('Val class counts: {}'.format([ CLASSES[idx] + ' ' + "{:.3f}".format(count) for idx, count in enumerate(class_counts) ] ))

	plt.bar([klass for klass in CLASSES], [acc for acc in class_accuracy], 1.0, color='#8F1500')
	axes = plt.gca()
	axes.set_ylim([0.01,1.0])
	plt.xticks(rotation='vertical')
	plt.show()

if __name__ == "__main__":
	main()
