from sklearn.externals import joblib

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

def testModel(tweet):
    print('Loading the Classifier, please wait....')
    classifier = joblib.load('saved_model/svmClassifier.pkl')
    print('READY')
    #tweet = 'مايشبه لوالو'
    print(predict(tweet, classifier))

def main():
    tweet = "هاذ الكتاب روعة اعجبني بزاف"
    testModel(tweet)

if __name__ == "__main__":
    main()
