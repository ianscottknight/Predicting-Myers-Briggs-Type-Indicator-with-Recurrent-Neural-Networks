import os
import numpy as np
import pandas as pd
import csv
import random
import pickle
import collections
import tensorflow as tf
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
from keras.optimizers import Adam


MODELS_DIR = "models"
DATA_DIR = "data"
GLOVE_PATH = os.path.join(DATA_DIR, "glove.6B.50d.txt")
DIMENSIONS = ["IE", "NS", "FT", "PJ"]

### Preprocessing variables
MODEL_BATCH_SIZE = 128
TOP_WORDS = 2500
MAX_POST_LENGTH = 40
EMBEDDING_VECTOR_LENGTH = 50

### Learning variables
LEARNING_RATE = 0.01
DROPOUT = 0.1
NUM_EPOCHS = 1

### Control variables
CROSS_VALIDATION = False
SAMPLE = True
WORD_CLOUD = True
SAVE_MODEL = True


for k in range(len(DIMENSIONS)):

    ###########################
    ### POST CLASSIFICATION ###
    ###########################

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    ### Read in data
    with open(
        os.path.join(DATA_DIR, "train_{}.csv".format(DIMENSIONS[k][0])), "r"
    ) as f:
        reader = csv.reader(f)
        for row in reader:
            for post in row:
                x_train.append(post)
                y_train.append(0)
    with open(
        os.path.join(DATA_DIR, "train_{}.csv".format(DIMENSIONS[k][1])), "r"
    ) as f:
        reader = csv.reader(f)
        for row in reader:
            for post in row:
                x_train.append(post)
                y_train.append(1)
    with open(os.path.join(DATA_DIR, "test_{}.csv".format(DIMENSIONS[k][0])), "r") as f:
        reader = csv.reader(f)
        for row in reader:
            for post in row:
                x_test.append(post)
                y_test.append(0)
    with open(os.path.join(DATA_DIR, "test_{}.csv".format(DIMENSIONS[k][1])), "r") as f:
        reader = csv.reader(f)
        for row in reader:
            for post in row:
                x_test.append(post)
                y_test.append(1)

    ### Preprocessing (lemmatization, tokenization, and padding of input)
    MBTI_TYPES = [
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
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokenizer = Tokenizer(num_words=TOP_WORDS, filters="")
    tokenizer.fit_on_texts(x_train + x_test)

    def lemmatize(x):
        lemmatized = []
        for post in x:
            temp = post.lower()
            for mbti_type in MBTI_TYPES:
                mbti_type = mbti_type.lower()
                temp = temp.replace(" " + mbti_type, "")
            temp = " ".join(
                [
                    lemmatizer.lemmatize(word)
                    for word in temp.split(" ")
                    if (word not in stop_words)
                ]
            )
            lemmatized.append(temp)
        return np.array(lemmatized)

    def preprocess(x):
        lemmatized = lemmatize(x)
        tokenized = tokenizer.texts_to_sequences(lemmatized)
        return sequence.pad_sequences(tokenized, maxlen=MAX_POST_LENGTH)

    x_train = lemmatize(x_train)
    x_test = lemmatize(x_test)

    ### Assign to dataframe and shuffle rows
    df = pd.DataFrame(data={"x": x_train, "y": y_train})
    df = df.sample(frac=1).reset_index(drop=True)  ### Shuffle rows
    if SAMPLE:
        df = df.head(10000)  ### Small sample for quick runs

    ### Load glove into memory for embedding
    embeddings_index = dict()
    with open(GLOVE_PATH) as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddings_index[word] = np.asarray(values[1:], dtype="float32")
    print("Loaded {} word vectors.".format(len(embeddings_index)))

    ### Create a weight matrix for words
    embedding_matrix = np.zeros((TOP_WORDS, EMBEDDING_VECTOR_LENGTH))
    for word, i in tokenizer.word_index.items():
        if i < TOP_WORDS:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    ### Construct model
    with tf.device("/gpu:0"):
        model = Sequential()
        model.add(
            Embedding(
                TOP_WORDS,
                EMBEDDING_VECTOR_LENGTH,
                input_length=MAX_POST_LENGTH,
                weights=[embedding_matrix],
                mask_zero=True,
                trainable=True,
            )
        )
        # model.add(SimpleRNN(EMBEDDING_VECTOR_LENGTH, dropout=DROPOUT, recurrent_dropout=DROPOUT, activation='sigmoid', kernel_initializer='zeros'))
        # model.add(GRU(EMBEDDING_VECTOR_LENGTH, dropout=DROPOUT, recurrent_dropout=DROPOUT, activation='sigmoid', kernel_initializer='zeros'))
        model.add(
            LSTM(
                EMBEDDING_VECTOR_LENGTH,
                dropout=DROPOUT,
                recurrent_dropout=DROPOUT,
                activation="sigmoid",
                kernel_initializer="zeros",
            )
        )
        # model.add(Bidirectional(LSTM(EMBEDDING_VECTOR_LENGTH, dropout=DROPOUT, recurrent_dropout=DROPOUT, activation='sigmoid', kernel_initializer='zeros')))
        model.add(Dense(1, activation="sigmoid"))
        optimizer = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        model.compile(
            loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )
        print(model.summary())

        ### Cross-validation classification (individual posts)
        if CROSS_VALIDATION:
            k_fold = KFold(n_splits=6)
            scores_k = []
            confusion_k = np.array([[0, 0], [0, 0]])
            for train_indices, test_indices in k_fold:
                x_train_k = df.iloc[train_indices]["x"].values
                y_train_k = df.iloc[train_indices]["y"].values
                x_test_k = df.iloc[test_indices]["x"].values
                y_test_k = df.iloc[test_indices]["y"].values
                model.fit(
                    preprocess(x_train_k),
                    y_train_k,
                    epochs=NUM_EPOCHS,
                    batch_size=MODEL_BATCH_SIZE,
                )
                predictions_k = model.predict_classes(preprocess(x_test_k))
                confusion_k += confusion_matrix(y_test_k, predictions_k)
                score_k = accuracy_score(y_test_k, predictions_k)
                scores_k.append(score_k)
            with open(
                os.path.join(
                    DATA_DIR, "rnn_cross_validation_{}.txt".format(DIMENSIONS[k])
                ),
                "w",
            ) as f:
                f.write(
                    "*** {}/{} TRAINING SET CROSS VALIDATION (POSTS) ***\n".format(
                        DIMENSIONS[k][0], DIMENSIONS[k][1]
                    )
                )
                f.write("Total posts classified: {}\n".format(len(x_train)))
                f.write("Accuracy: {}\n".format(sum(scores_k) / len(scores_k)))
                f.write("Confusion matrix: \n")
                f.write(np.array2string(confusion_k, separator=", "))

        ### Test set classification (individual posts)
        model.fit(
            preprocess(df["x"].values),
            df["y"].values,
            epochs=NUM_EPOCHS,
            batch_size=MODEL_BATCH_SIZE,
        )
        predictions = model.predict_classes(preprocess(x_test))
        confusion = confusion_matrix(y_test, predictions)
        score = accuracy_score(y_test, predictions)
        with open(
            os.path.join(MODELS_DIR, "rnn_accuracy_{}.txt".format(DIMENSIONS[k])), "w"
        ) as f:
            f.write(
                "*** {}/{} TEST SET CLASSIFICATION (POSTS) ***\n".format(
                    DIMENSIONS[k][0], DIMENSIONS[k][1]
                )
            )
            f.write("Total posts classified: {}\n".format(len(x_test)))
            f.write("Accuracy: {}\n".format(score))
            f.write("Confusion matrix: \n")
            f.write(np.array2string(confusion, separator=", "))
        print(
            f"\nWrote training / test results for {DIMENSIONS[k]} here: {os.path.join(MODELS_DIR, 'rnn_accuracy_{}.txt'.format(DIMENSIONS[k]))}\n"
        )

        ### Get most a-like/b-like sentences
        if WORD_CLOUD:
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
            with open(
                os.path.join(
                    DATA_DIR, "extreme_examples_{}.txt".format(DIMENSIONS[k][0])
                ),
                "w",
            ) as f:
                for prob, i in min_prob_indices:
                    # f.write(x_test[i]+'\n')
                    f.write(x_test[i] + "\n")
                    # f.write(str(prob)+'\n')
                    f.write("\n")
            with open(
                os.path.join(
                    DATA_DIR, "extreme_examples_{}.txt".format(DIMENSIONS[k][1])
                ),
                "w",
            ) as f:
                for prob, i in max_prob_indices:
                    # f.write(x_test[i]+'\n')
                    f.write(x_test[i] + "\n")
                    # f.write(str(prob)+'\n')
                    f.write("\n")

        ### Save model and tokenizer for future use
        model.save(os.path.join(MODELS_DIR, "rnn_model_{}.h5".format(DIMENSIONS[k])))
        with open(
            os.path.join(MODELS_DIR, "rnn_tokenizer_{}.pkl".format(DIMENSIONS[k])), "wb"
        ) as f:
            pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
