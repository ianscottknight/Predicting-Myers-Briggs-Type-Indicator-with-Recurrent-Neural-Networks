import os
import collections
import pandas as pd
import csv
import re


DATA_DIR = "data"
MBTI_CLEAN_CSV_PATH = os.path.join(DATA_DIR, "mbti_clean.csv")
DIMENSIONS = ("IE", "NS", "TF", "PJ")


df = pd.read_csv(MBTI_CLEAN_CSV_PATH)

for dimension in DIMENSIONS:
    letter_1, letter_2 = dimension
    for letter in [letter_1, letter_2]:
        posts = []
        for index, row in df.iterrows():
            if letter in row["type"]:
                hundred_posts = row["posts"].split("|||")
                for post in hundred_posts:
                    if (
                        ("http" in post)
                        or (post == "")
                        or (post == None)
                        or (not re.search("[a-zA-Z]", post))
                    ):  # ignore deformed posts
                        continue
                    posts.append(post)

        test_csv_path = os.path.join(DATA_DIR, f"test_{letter}.csv")
        with open(test_csv_path, "w") as f:
            writer = csv.writer(f)
            for post in posts:
                writer.writerow([post])
