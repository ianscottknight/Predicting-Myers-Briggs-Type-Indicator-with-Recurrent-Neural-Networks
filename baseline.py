#!/usr/bin/env python

import numpy as np
import pandas as pd
import csv
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB 
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 

DIMENSIONS = ['IE', 'NS', 'FT', 'PJ']
CROSS_VALIDATION = False
SAVE_MODEL = False

for k in range(len(DIMENSIONS)):
	x_train = [] 
	y_train = [] 
	x_test = [] 
	y_test = []

	### Read in data
	with open('train_{}.csv'.format(DIMENSIONS[k][0]), 'r') as f: 
		reader = csv.reader(f)
		for row in reader:
			for post in row: 
				x_train.append(post)
				y_train.append(0)
	with open('train_{}.csv'.format(DIMENSIONS[k][1]), 'r') as f: 
		reader = csv.reader(f)
		for row in reader:
			for post in row: 
				x_train.append(post)
				y_train.append(1)
	with open('test_{}.csv'.format(DIMENSIONS[k][0]), 'r') as f: 
		reader = csv.reader(f)
		for row in reader:
			for post in row: 
				x_test.append(post)
				y_test.append(0)
	with open('test_{}.csv'.format(DIMENSIONS[k][1]), 'r') as f: 
		reader = csv.reader(f)
		for row in reader:
			for post in row: 
				x_test.append(post)
				y_test.append(1)

	types = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP', 'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
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
	def preprocess(x):
		lemmatized = lemmatize(x)
		tokenized = tokenizer.texts_to_sequences(lemmatized)
		return sequence.pad_sequences(tokenized, maxlen=MAX_POST_LENGTH)

	x_train = lemmatize(x_train)
	x_test = lemmatize(x_test)

	### Assign to dataframe
	df = pd.DataFrame(data={'text': x_train, 'type': y_train})
	df = df.sample(frac=1).reset_index(drop=True) ### Shuffle rows

	'''
	### Make pipeline
	pipeline = Pipeline([
		('vectorizer',  CountVectorizer(stop_words='english')), ### Bag-of-words
		('transformer', TfidfTransformer()), 
		('classifier',  MultinomialNB()) ]) ### Performs best
		#('classifier',  SVC()) ])
		#('classifier',  DecisionTreeClassifier(max_depth=50)) ])
		#('classifier', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)) ])
		#('classifier',  Perceptron()) ])
		#('classifier',  KNeighborsClassifier(n_neighbors=2)) ])
	'''

	### Cross-validation classification (individual posts)
	if CROSS_VALIDATION: 
		k_fold = KFold(n=len(df), n_folds=6)
		scores_k = []
		confusion_k = np.array([[0, 0], [0, 0]])
		for train_indices, test_indices in k_fold:
			x_train_k = df.iloc[train_indices]['text'].values
			y_train_k = df.iloc[train_indices]['type'].values
			x_test_k = df.iloc[test_indices]['text'].values
			y_test_k = df.iloc[test_indices]['type'].values
			pipeline.fit(x_train_k, y_train_k)
			predictions_k = pipeline.predict(x_test_k)
			confusion_k += confusion_matrix(y_test_k, predictions_k)
			score_k = accuracy_score(y_test_k, predictions_k)
			scores_k.append(score_k)
		with open('cross_validation_{}.txt'.format(DIMENSIONS[k]), 'w') as f: 
			f.write('*** {}/{} TRAINING SET CROSS VALIDATION (POSTS) ***\n'.format(DIMENSIONS[k][0], DIMENSIONS[k][1]))
			f.write('Total posts classified: {}\n'.format(len(df)))
			f.write('Accuracy: {}\n'.format(sum(scores_k)/len(scores_k)))
			f.write('Confusion matrix: \n')
			f.write(np.array2string(confusion_k, separator=', '))

	### Test set classification (individual posts)
	pipeline.fit(df['text'].values, df['type'].values)
	predictions = pipeline.predict(x_test) 
	confusion = confusion_matrix(y_test, predictions)
	score = accuracy_score(y_test, predictions)
	with open('test_set_post_classification_{}.txt'.format(DIMENSIONS[k]), 'w') as f: 
		f.write('*** {}/{} TEST SET CLASSIFICATION (POSTS) ***\n'.format(DIMENSIONS[k][0], DIMENSIONS[k][1]))
		f.write('Total posts classified: {}\n'.format(len(x_test)))
		f.write('Accuracy: {}\n'.format(score))
		f.write('Confusion matrix: \n')
		f.write(np.array2string(confusion, separator=', '))

	###########################
	### USER CLASSIFICATION ###
	###########################
	
	### Read in user test set data
	x_test_a = []
	x_test_b = []
	with open('test_{}.csv'.format(DIMENSIONS[k][0]), 'r') as f: 
		reader = csv.reader(f)
		for row in reader:
			x_test_a.append(row)
	with open('test_{}.csv'.format(DIMENSIONS[k][1]), 'r') as f: 
		reader = csv.reader(f)
		for row in reader: 
			x_test_b.append(row)
	x_test = x_test_a + x_test_b

	### Make predictions for users
	'''
	predictions_a = []
	for user_batch in x_test_a:
		predictions_for_batch = pipeline.predict_classes(preprocess(user_batch))
		predictions_for_batch = [item for sublist in predictions_for_batch for item in sublist] ### Make flat list
		prediction = collections.Counter(predictions_for_batch).most_common(1)[0][0]
		predictions_a.append(prediction)
	predictions_b = []
	for user_batch in x_test_b:
		predictions_for_batch = pipeline.predict_classes(preprocess(user_batch))
		predictions_for_batch = [item for sublist in predictions_for_batch for item in sublist] ### Make flat list
		prediction = collections.Counter(predictions_for_batch).most_common(1)[0][0]
		predictions_b.append(prediction)
	predictions = predictions_a + predictions_b
	'''

	predictions = []
	for user in x_test: 
		prob = float(sum(pipeline.predict_proba(preprocess(user)))/len(user))
		prediction = 1
		if prob < 0.5: 
			prediction = 0
		predictions.append(prediction)

	### Analyze user classification results
	y_test_a = [0 for ___ in predictions_a]
	y_test_b = [1 for ___ in predictions_b]
	y_test = y_test_a + y_test_b
	confusion = confusion_matrix(y_test, predictions)
	score = accuracy_score(y_test, predictions)
	with open('test_set_user_classification_{}.txt'.format(DIMENSIONS[k]), 'w') as f: 
		f.write('*** {}/{} TEST SET CLASSIFICATION (USERS) ***\n'.format(DIMENSIONS[k][0], DIMENSIONS[k][1]))
		f.write('Total posts classified: {}\n'.format(len(x_test)))
		f.write('Accuracy: {}\n'.format(score))
		f.write('Confusion matrix: \n')
		f.write(np.array2string(confusion, separator=', '))

	# Save model
	if SAVE_MODEL:
		pipeline.named_steps['classifier'].model.save('NB_classifier_{}.h5'.format(DIMENSIONS[k]))
		pipeline.named_steps['classifier'].model = None
		joblib.dump(pipeline, 'NB_pipeline_{}.pkl'.format(DIMENSIONS[k]))
	del pipeline
