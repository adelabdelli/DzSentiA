import gensim 
from gensim.models import KeyedVectors
import numpy as np
from gensim.models import word2vec

#Download the word2vec corpus from here : https://drive.google.com/open?id=1yDxUsn9FqM7OQqubuoaabO9BWDszcqpS
sentences = gensim.models.word2vec.LineSentence('word2vec_corpus.txt')
print("Text loaded!")
model = word2vec.Word2Vec(sentences, size=300,window=9,min_count=10,workers= 16)
print("Model trained!")
model.save()
model.wv.save_word2vec_format('model.bin')
X = model[model.wv.vocab]
words = list(model.wv.vocab)
print("number of words is : ",len(words))
np.save('wordsList.npy',words)
np.save('wordVectors.npy',X)

