import os
import collections
import re


DATA_DIR = "data"

DIMENSIONS = ["IE", "NS", "FT", "PJ"]
WORDS_TO_REMOVE = [
    "intj",
    "intp",
    "infj",
    "infp",
    "istj",
    "istp",
    "isfj",
    "isfp",
    "entj",
    "entp",
    "enfj",
    "enfp",
    "estj",
    "estp",
    "esfj",
    "esfp",
    "si",
    "ni",
    "ti",
    "fi",
    "se",
    "ne",
    "te",
    "fe",
    "nt",
    "nf",
    "sxsp",
    "spsx",
    "spso",
    "sxso",
    "sosp",
    "sosx",
    "sp",
    "sx",
    "sj",
    "sf",
    "st",
    "le",
    "socionic",
    "socionics",
    "enneagram",
    "d",
    "w",
    "mbti",
]

for k in range(len(DIMENSIONS)):

    wordcount_a = {}
    wordcount_b = {}

    with open(
        os.path.join(DATA_DIR, "extreme_examples_{}.txt".format(DIMENSIONS[k][0])), "r"
    ) as f:
        wordcount_a = collections.Counter(f.read().split())

    with open(
        os.path.join(DATA_DIR, "extreme_examples_{}.txt".format(DIMENSIONS[k][1])), "r"
    ) as f:
        wordcount_b = collections.Counter(f.read().split())

    cache = []

    for key in wordcount_a.keys():
        if key not in cache:
            cache.append(key)
    for key in wordcount_b.keys():
        if key not in cache:
            cache.append(key)

    a = {}
    b = {}

    for key in cache:
        if key in wordcount_a.keys():
            if key in wordcount_b.keys():
                if wordcount_a[key] > wordcount_b[key]:
                    a[key] = wordcount_a[key] - wordcount_b[key]
                elif wordcount_a[key] < wordcount_b[key]:
                    b[key] = wordcount_b[key] - wordcount_a[key]
            else:
                a[key] = wordcount_a[key]
        elif key in wordcount_b.keys():
            b[key] = wordcount_b[key]

    regex = re.compile("[^a-zA-Z]")

    with open(
        os.path.join(DATA_DIR, "special_words_{}.txt".format(DIMENSIONS[k][0])), "w"
    ) as f:
        for key in a.keys():
            mod = regex.sub("", str(key))
            if mod not in WORDS_TO_REMOVE:
                if a[key] > 2:
                    for ___ in range(a[key]):
                        f.write(mod + "\n")

    with open(
        os.path.join(DATA_DIR, "special_words_{}.txt".format(DIMENSIONS[k][1])), "w"
    ) as f:
        for key in b.keys():
            mod = regex.sub("", str(key))
            if mod not in WORDS_TO_REMOVE:
                if b[key] > 2:
                    for ___ in range(b[key]):
                        f.write(mod + "\n")
