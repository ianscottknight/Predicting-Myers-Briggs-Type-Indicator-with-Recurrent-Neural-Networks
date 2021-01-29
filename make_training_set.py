import os
import collections
import pandas as pd
import csv
import re


DATA_DIR = "data"
MBTI_UNCLEAN_CSV_PATH = os.path.join(DATA_DIR, "mbti_unclean.csv")
DIMENSIONS = ("IE", "NS", "TF", "PJ")


df = pd.read_csv(MBTI_UNCLEAN_CSV_PATH)

counts = collections.defaultdict(int)
for dimension in DIMENSIONS:
    letter_1, letter_2 = dimension
    for index, row in df.iterrows():
        mbti = row["type"]
        hundred_posts = row["posts"].split("|||")
        for post in hundred_posts:
            if (
                ("http" in post)
                or (post == "")
                or (post == None)
                or (not re.search("[a-zA-Z]", post))
            ):  # ignore deformed posts
                continue
            if letter_1 in mbti:
                counts[letter_1] += 1
            if letter_2 in mbti:
                counts[letter_2] += 1

for dimension in DIMENSIONS:
    letter_1, letter_2 = dimension
    if counts[letter_1] < counts[letter_2]:
        limit = counts[letter_1]
    else:
        limit = counts[letter_2]

    for letter in [letter_1, letter_2]:
        posts = []
        i = 0
        for index, row in df.iterrows():
            if letter in row["type"]:
                hundred_posts = row["posts"].split("|||")
                for post in hundred_posts:
                    if i == limit:
                        break
                    if (
                        ("http" in post)
                        or (post == "")
                        or (post == None)
                        or (not re.search("[a-zA-Z]", post))
                    ):  # ignore deformed posts
                        continue
                    posts.append(post)
                    i += 1

        train_csv_path = os.path.join(DATA_DIR, f"train_{letter}.csv")
        with open(train_csv_path, "w") as f:
            writer = csv.writer(f)
            for post in posts:
                writer.writerow([post])
