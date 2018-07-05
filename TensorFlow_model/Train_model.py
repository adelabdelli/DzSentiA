#!/usr/bin/env python
#-*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
import datetime
from random import randint
import time

#parameters
maxSeqLength = 250
batchSize = 20
lstmUnits = 64
numClasses = 2
iterations = 100000
numDimensions = 300 
wordsList = np.load('../Word2Vec/wordsList.npy')
wordsList = wordsList.tolist() 
wordVectors = np.load('../Word2Vec/wordVectors.npy')



def readData(filename):

    data = csv.reader(open(filename, 'r'), delimiter=",", quotechar='|')
    text, X = [], []
    numWords = []

    for row in data:
        text.append(row[1])

    # to delete the first row of cvs file (the title)
    for i in range(1,len(text)):
        X.append(text[i])

    for i in range(len(X)):
        counter = len(X[i].split())
        numWords.append(counter)

    numFiles = len(numWords)
	
    return X,numWords,numFiles


def build_idsMatrix(X,numFiles):
    ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
    fileCounter = 0
    f = open("unknown_words.txt","a",encoding="utf-8")
    for i in range(len(X)):
        indexCounter = 0
        line = X[i]
        split = line.split()
        split = split[::-1] #inverse the words because arabic is written from right to left
        for word in split:
            try:
                if indexCounter <250: # 250 is the max sequence length
                    ids[fileCounter][indexCounter] = wordsList.index(word)
            except ValueError:
                ids[fileCounter][indexCounter] = 0 #Vector for unkown words
                f.write(str(word+"\n"))

            indexCounter = indexCounter + 1

            if indexCounter >= maxSeqLength:
                break
        fileCounter = fileCounter + 1
    np.save('idsMatrix', ids)


def getTrainBatch():

    ids = np.load('idsMatrix.npy')
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0):
            num = randint(28672, 49864)
            labels.append([1,0])
        else:
            num = randint(1, 21192)
            labels.append([0,1])
        arr[i] = ids[num-1:num]

    return arr, labels

def getTestBatch():

    ids = np.load('idsMatrix.npy')
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(21192, 28672)
        if (num <= 24932):
            labels.append([0,1])
        else:
            labels.append([1,0])
        arr[i] = ids[num-1:num]

    return arr, labels

def trainModel():

    sess = tf.Session()
    tf.reset_default_graph()

    labels = tf.placeholder(tf.float32, [batchSize, numClasses])
    input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

    data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype=tf.float32)
    data = tf.nn.embedding_lookup(wordVectors, input_data)

    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
    value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)

    correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge_all()
    logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir, sess.graph)

    t = []
    t.append(time.time())
    #start training
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())


    for i in range(iterations):
        # Next Batch of reviews
        nextBatch, nextBatchLabels = getTrainBatch();
        sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})
        print("i = ", i)
        # Write summary to Tensorboard
        if (i % 50 == 0):
            summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
            writer.add_summary(summary, i)

        # Save the network every 10,000 training iterations
        if (i % 10000 == 0 and i != 0):
            tt = time.time()
            print("t",i," : ",tt)
            t.append(tt)
            save_path = saver.save(sess, "saved_model/pretrained_lstm.ckpt", global_step=i)
            print("saved to %s" % save_path)
    writer.close()
    t.append(time.time())
    print("time taken : ", t[len(t)-1]-t[0])
    print("t vector \n",t)


def main():
    X, numWords, numFiles = readData('../dataset.csv')
    build_idsMatrix(X, numFiles)
    trainModel()

if __name__ == "__main__":
    main()