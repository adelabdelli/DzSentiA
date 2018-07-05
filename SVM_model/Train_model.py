
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

import csv
from sklearn.metrics import confusion_matrix
import time

def readData(filename):

    data = csv.reader(open(filename, 'r'), delimiter=",", quotechar='|')
    text, sentiment,X, y = [], [],[],[]

    for row in data:
        text.append(row[1])
        sentiment.append(row[2])

    for i in range(1,len(text)):
        X.append(text[i])
        if(sentiment[i] == 'Negative'):
            y.append(0)
        else:
            y.append(1)

    return X,y

def getTrainingAndTestData(X,y):

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.15, random_state=42)

    return X_train, X_test, y_train, y_test

def classifier(X_train, y_train):

    vec = TfidfVectorizer(min_df=5, max_df=0.95, sublinear_tf=True, use_idf=True, ngram_range=(1, 2))
    svm_clf = svm.LinearSVC(C=0.1)
    vec_clf = Pipeline([('vectorizer', vec), ('pac', svm_clf)])
    vec_clf.fit(X_train, y_train)
    joblib.dump(vec_clf, 'saved_model/svmClassifier.pkl', compress=3)

    return vec_clf


def predict(tweet, classifier):
    if (('__positive__') in (tweet)):
        sentiment = 1
        return sentiment

    elif (('__negative__') in (tweet)):
        sentiment = 0
        return sentiment
    else:

        X = [tweet]
        sentiment = classifier.predict(X)
    return (sentiment[0])



def main():
    X,y = readData('../dataset.csv')
    X_train, X_test, y_train, y_test = getTrainingAndTestData(X,y)
    t1 = time.time()
    vec_clf = classifier(X_train, y_train)
    y_pred = vec_clf.predict(X_test)
    t2 = time.time()
    print("time taken : ",t2-t1)
    print(sklearn.metrics.classification_report(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("true negative :",tn," true positive",tp,"\n false negative",fn," false positive",fp)
if __name__ == "__main__":
    main()