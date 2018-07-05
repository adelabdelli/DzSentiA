#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from random import randint
from sklearn.metrics import confusion_matrix

#parameters
maxSeqLength = 250
batchSize = 20
lstmUnits = 64
numClasses = 2
iterations = 10002
numDimensions = 300 

wordsList = np.load('../Word2Vec/wordsList.npy').tolist()
wordVectors = np.load('../Word2Vec/wordVectors.npy')

randList = []

def getTestBatch():

    ids = np.load('idsMatrix.npy')
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(20000, 28672) 
        randList.append(num)
        if (num <= 24932):
            labels.append([0,1])
        else:
            labels.append([1,0])
        arr[i] = ids[num-1:num]

    return arr, labels

def getSentenceMatrix(sentence):
    arr = np.zeros([batchSize, maxSeqLength])
    sentenceMatrix = np.zeros([batchSize,maxSeqLength], dtype='int32')
    split = sentence.split()
    print(split)
    split = split[::-1]
    print(split)
    for indexCounter,word in enumerate(split):
        try:
            sentenceMatrix[0,indexCounter] = wordsList.index(word)
        except ValueError:
            sentenceMatrix[0,indexCounter] = 0 #Vector for unkown words
    return sentenceMatrix

def predict(inputText):


    inputMatrix = getSentenceMatrix(inputText)
    #prepare the model
    tf.reset_default_graph()

    labels = tf.placeholder(tf.float32, [batchSize, numClasses])
    input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

    data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype=tf.float32)
    data = tf.nn.embedding_lookup(wordVectors, input_data)

    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.25)
    value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)


    correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('saved_model/'))


    iterations = 374

    tnS,tpS,fpS,fnS = 0,0,0,0
    sum = 0
    for i in range(iterations):
        nextBatch, nextBatchLabels = getTestBatch()
        sum += (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100
        print("Accuracy for this batch i: ",i," ", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)
        cprrectPNpy = sess.run(labels, {input_data: nextBatch, labels: nextBatchLabels})
        predictionNpy = sess.run(prediction, {input_data: nextBatch, labels: nextBatchLabels})
        #print(cprrectPNpy)
        cprrectPNpyBool = []
        for i in range(len(cprrectPNpy)):
            if cprrectPNpy[i][0] == 1. :
                cprrectPNpyBool.append(True)
            else:
                cprrectPNpyBool.append(False)

        predictionNpyBool = []
        predictionNpyList = predictionNpy.tolist()
        for i in range(len(predictionNpyList)):
            if predictionNpyList[i][0] > predictionNpyList[i][1]:
                predictionNpyBool.append(True)
            else:
                predictionNpyBool.append(False)
        tn, fp, fn, tp = confusion_matrix(cprrectPNpyBool, predictionNpyBool).ravel()
        tnS+=tn
        fpS+=fp
        fnS+=fn
        tpS+=tp



    print("Accuracy : ",sum/iterations)
    predictedSentiment = sess.run(prediction, {input_data: inputMatrix})[0]
    predictionNpyBool = []
    predictionNpyList = predictionNpy.tolist()

    for i in range(len(predictionNpyList)):
        if predictionNpyList[i][0]> predictionNpyList[i][1]:
            predictionNpyBool.append(True)
        else:
            predictionNpyBool.append(False)

    print("true negatif : ",tnS, " false positif : ", fpS, " false negatif : ", fnS, " true positive : ", tpS)
    print("accuracy : ",(tpS+tnS)/(tnS+tpS+fpS+fnS))
    print("precision : ", tpS/(tpS+fpS))

    if (predictedSentiment[0] > predictedSentiment[1]):
        print ("Positive Sentiment")
    else:
        print ("Negative Sentiment")
		
def main():
    s = "الكتاب عجبني بزاف"
    #s = "تطور الاقتصاد الجزائري"
    predict(s)

if __name__ == "__main__":
    main()