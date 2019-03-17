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
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt

import numpy as np

TRAIN_PATH = 'data/processed/train.csv'
VAL_PATH = 'data/processed/valid.csv'
DELIMITER = '|'
CLASSES = ['mets', 'memory', 'multiple sclerosis', 'epilepsy', 'stereo/cyberknife', 'routine brain', 'sella', 'tumor brain', 'complex headache', 'brain trauma', 'stroke']

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
        if pred == int(y[idx]):
            class_correct[pred] += 1
        class_counts[int(y[idx])] += 1

    return class_correct, class_counts

def trainModel(model):
    class_correct = np.zeros(len(CLASSES))
    class_counts = np.zeros(len(CLASSES))


    train_X, train_Y = readInData(TRAIN_PATH)
    valid_X, valid_Y = readInData(VAL_PATH)

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

    print('Accuracy on one fold = {}'.format(accuracy))


    fold_class_correct, fold_class_counts = class_accuracy(predictions, valid_Y)
    precision, recall, f_score, _ = precision_recall_fscore_support(valid_Y, predictions)

    print(fold_class_correct)

    class_correct += fold_class_correct
    class_counts += fold_class_counts

    print('Completed a kfold')

    return accuracy, recall, precision, f_score, class_correct, class_counts

def main():

    ## Use comments to choose a particular model to train
    clf = MultinomialNB()
    # clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    # clf = SGDClassifier(random_state=314159)

    ## KFold training
    kfold_accuracy, kfold_recall, precision, f_score, class_correct, class_counts = trainModel(clf)
    print('Overall KFold accuracy was {} and recall was {}'.format(kfold_accuracy, kfold_recall))

    # I have given up on style
    print(precision)
    print(f_score)

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
