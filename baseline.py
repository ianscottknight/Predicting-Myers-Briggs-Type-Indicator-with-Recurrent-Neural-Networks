import os
import numpy as np
import pandas as pd
import csv
import joblib
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


MODELS_DIR = "models"
DATA_DIR = "data"
DIMENSIONS = ["IE", "NS", "FT", "PJ"]
TOP_WORDS = 2500
MAX_POST_LENGTH = 40
CROSS_VALIDATION = False
SAVE_MODEL = False

for k in range(len(DIMENSIONS)):
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

    ### Assign to dataframe
    df = pd.DataFrame(data={"text": x_train, "type": y_train})
    df = df.sample(frac=1).reset_index(drop=True)  ### Shuffle rows

    ### Make pipeline
    pipeline = Pipeline(
        [
            ("vectorizer", CountVectorizer(stop_words="english")),  ### Bag-of-words
            ("transformer", TfidfTransformer()),
            ("classifier", MultinomialNB()),
        ]
    )  ### Performs best
    # ('classifier',  SVC()) ])
    # ('classifier',  DecisionTreeClassifier(max_depth=50)) ])
    # ('classifier', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)) ])
    # ('classifier',  Perceptron()) ])
    # ('classifier',  KNeighborsClassifier(n_neighbors=2)) ])

    ### Cross-validation classification (individual posts)
    if CROSS_VALIDATION:
        k_fold = KFold(n_splits=6)
        scores_k = []
        confusion_k = np.array([[0, 0], [0, 0]])
        for train_indices, test_indices in k_fold:
            x_train_k = df.iloc[train_indices]["text"].values
            y_train_k = df.iloc[train_indices]["type"].values
            x_test_k = df.iloc[test_indices]["text"].values
            y_test_k = df.iloc[test_indices]["type"].values
            pipeline.fit(x_train_k, y_train_k)
            predictions_k = pipeline.predict(x_test_k)
            confusion_k += confusion_matrix(y_test_k, predictions_k)
            score_k = accuracy_score(y_test_k, predictions_k)
            scores_k.append(score_k)
        with open(
            os.path.join(
                MODELS_DIR, "baseline_cross_validation_{}.txt".format(DIMENSIONS[k])
            ),
            "w",
        ) as f:
            f.write(
                "*** {}/{} TRAINING SET CROSS VALIDATION (POSTS) ***\n".format(
                    DIMENSIONS[k][0], DIMENSIONS[k][1]
                )
            )
            f.write("Total posts classified: {}\n".format(len(df)))
            f.write("Accuracy: {}\n".format(sum(scores_k) / len(scores_k)))
            f.write("Confusion matrix: \n")
            f.write(np.array2string(confusion_k, separator=", "))

    ### Test set classification (individual posts)
    pipeline.fit(df["text"].values, df["type"].values)
    predictions = pipeline.predict(x_test)
    confusion = confusion_matrix(y_test, predictions)
    score = accuracy_score(y_test, predictions)
    with open(
        os.path.join(MODELS_DIR, "baseline_accuracy_{}.txt".format(DIMENSIONS[k])), "w"
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
        f"Wrote training / test results for {DIMENSIONS[k]} here: {os.path.join(MODELS_DIR, 'baseline_accuracy_{}.txt'.format(DIMENSIONS[k]))}"
    )

    # Save model
    if SAVE_MODEL:
        pipeline.named_steps["classifier"].model.save(
            os.path.join(MODELS_DIR, "NB_classifier_{}.h5".format(DIMENSIONS[k]))
        )
        pipeline.named_steps["classifier"].model = None
        joblib.dump(
            pipeline,
            os.path.join(MODELS_DIR, "baseline_pipeline_{}.pkl".format(DIMENSIONS[k])),
        )
    del pipeline
