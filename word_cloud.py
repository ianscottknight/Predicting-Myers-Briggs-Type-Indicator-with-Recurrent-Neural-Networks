import os
import numpy as np
import pandas as pd
import csv
import random
import pickle
import collections
import tensorflow as tf
from nltk import word_tokenize
from keras.preprocessing.text import Tokenizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import GRU
from keras.layers import SimpleRNN
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing import text
from keras.models import load_model


MODELS_DIR = "models"
DATA_DIR = "data"

DIMENSIONS = ["IE", "NS", "FT", "PJ"]
MODEL_BATCH_SIZE = 128
TOP_WORDS = 2500
MAX_POST_LENGTH = 40
EMBEDDING_VECTOR_LENGTH = 50

types = [
    "INFJ",
    "ENTP",
    "INTP",
    "INTJ",
    "ENTJ",
    "ENFJ",
    "INFP",
    "ENFP",
    "ISFP",
    "ISTP",
    "ISFJ",
    "ISTJ",
    "ESTP",
    "ESFP",
    "ESTJ",
    "ESFJ",
]
types = [x.lower() for x in types]
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words("english")


def lemmatize(x):
    lemmatized = []
    for user in x:
        for post in user:
            temp = post.lower()
            for type_ in types:
                temp = temp.replace(" " + type_, "")
            temp = " ".join(
                [
                    lemmatizer.lemmatize(word)
                    for word in temp.split(" ")
                    if (word not in stop_words)
                ]
            )
            lemmatized.append(temp)
    return np.array(lemmatized)


for k in range(len(DIMENSIONS)):
    x_test_a = []
    x_test_b = []
    with open(os.path.join(DATA_DIR, "test_{}.csv".format(DIMENSIONS[k][0])), "r") as f:
        reader = csv.reader(f)
        for row in reader:
            x_test_a.append(row)
    with open(os.path.join(DATA_DIR, "test_{}.csv".format(DIMENSIONS[k][1])), "r") as f:
        reader = csv.reader(f)
        for row in reader:
            x_test_b.append(row)
    x_test = x_test_a + x_test_b

    model = load_model(
        os.path.join(MODELS_DIR, "rnn_model_{}.h5".format(DIMENSIONS[k]))
    )
    tokenizer = None
    with open(
        os.path.join(MODELS_DIR, "rnn_tokenizer_{}.pkl".format(DIMENSIONS[k])), "rb"
    ) as f:
        tokenizer = pickle.load(f)

    def preprocess(x):
        lemmatized = lemmatize(x)
        tokenized = tokenizer.texts_to_sequences(lemmatized)
        return sequence.pad_sequences(tokenized, maxlen=MAX_POST_LENGTH)

    NUM_EXTREME_EXAMPLES = 500
    probs = model.predict_proba(preprocess(x_test))
    scores = []
    indices = []
    for i, prob in enumerate(probs, 0):
        scores.append(prob[0])
        indices.append(i)
    sorted_probs = sorted(zip(scores, indices))
    min_prob_indices = sorted_probs[:NUM_EXTREME_EXAMPLES]
    max_prob_indices = sorted_probs[-NUM_EXTREME_EXAMPLES:]
    lemmatized = lemmatize(x_test)
    with open(
        os.path.join(DATA_DIR, "extreme_examples_{}.txt".format(DIMENSIONS[k][0])), "w"
    ) as f:
        for prob, i in min_prob_indices:
            # f.write(x_test[i]+'\n')
            f.write(lemmatized[i] + "\n")
            # f.write(str(prob)+'\n')
            f.write("\n")
    with open(
        os.path.join(DATA_DIR, "extreme_examples_{}.txt".format(DIMENSIONS[k][1])), "w"
    ) as f:
        for prob, i in max_prob_indices:
            # f.write(x_test[i]+'\n')
            f.write(lemmatized[i] + "\n")
            # f.write(str(prob)+'\n')
            f.write("\n")
