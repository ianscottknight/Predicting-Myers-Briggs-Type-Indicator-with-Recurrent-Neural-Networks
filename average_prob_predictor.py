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
from sklearn.metrics import confusion_matrix, accuracy_score

DIMENSIONS = ['IE', 'NS', 'FT', 'PJ']
MAX_POST_LENGTH = 40

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

for k in range(len(DIMENSIONS)):
    ### Read in data
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

    model = load_model('model_{}.h5'.format(DIMENSIONS[k]))
    tokenizer = None
    with open('tokenizer_{}.pkl'.format(DIMENSIONS[k]), 'rb') as f:
        tokenizer = pickle.load(f)
    def preprocess(x):
        lemmatized = lemmatize(x)
        tokenized = tokenizer.texts_to_sequences(lemmatized)
        return sequence.pad_sequences(tokenized, maxlen=MAX_POST_LENGTH)
    predictions = []
    for user in x_test: 
        prob = float(sum(model.predict_proba(preprocess(user)))/len(user))
        prediction = 1
        if prob < 0.5: 
            prediction = 0
        predictions.append(prediction)

    ### Analyze results
    y_test_a = [0 for ___ in x_test_a]
    y_test_b = [1 for ___ in x_test_b]
    y_test = y_test_a + y_test_b
    confusion = confusion_matrix(y_test, predictions)
    score = accuracy_score(y_test, predictions)
    with open('test_set_user_classification_{}.txt'.format(DIMENSIONS[k]), 'w') as f: 
        f.write('*** {}/{} TEST SET CLASSIFICATION (USERS) ***\n'.format(DIMENSIONS[k][0], DIMENSIONS[k][1]))
        f.write('Total posts classified: {}\n'.format(len(x_test)))
        f.write('Accuracy: {}\n'.format(score))
        f.write('Confusion matrix: \n')
        f.write(np.array2string(confusion, separator=', '))
