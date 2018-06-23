#!/usr/bin/env python

import csv
import pickle
import collections
import numpy as np
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from keras.preprocessing import sequence
from keras.preprocessing import text
from keras.models import load_model

DIMENSIONS = ['IE', 'NS', 'FT', 'PJ']
MODEL_BATCH_SIZE = 128
TOP_WORDS = 2500
MAX_POST_LENGTH = 40
EMBEDDING_VECTOR_LENGTH = 20

final = ''

x_test = []
with open('trumptweets.csv', 'r', encoding="ISO-8859-1") as f:
	reader = csv.reader(f)
	for row in f:
		x_test.append(row)

types = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
	 'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
types = [x.lower() for x in types]
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words("english")

def lemmatize(x):
	lemmatized = []
	for post in x: 
		temp = post.lower() 
		for type_ in types: 
			temp = temp.replace(' ' + type_, '')
		temp = ' '.join([lemmatizer.lemmatize(word) for word in temp.split(' ') if (word not in stop_words)])
		lemmatized.append(temp)
	return np.array(lemmatized)

for k in range(len(DIMENSIONS)):
	model = load_model('model_{}.h5'.format(DIMENSIONS[k]))
	tokenizer = None
	with open('tokenizer_{}.pkl'.format(DIMENSIONS[k]), 'rb') as f:
		tokenizer = pickle.load(f)
	def preprocess(x):
		lemmatized = lemmatize(x)
		tokenized = tokenizer.texts_to_sequences(lemmatized)
		return sequence.pad_sequences(tokenized, maxlen=MAX_POST_LENGTH)
	predictions = model.predict(preprocess(x_test))
	prediction = float(sum(predictions)/len(predictions))
	print(DIMENSIONS[k])
	print(prediction)
	if prediction >= 0.5: 
		final += DIMENSIONS[k][1]
	else: 
		final += DIMENSIONS[k][0]

print('')
print('Final prediction: {}'.format(final))

