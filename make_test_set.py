#!/usr/bin/env python

import random
import collections
import numpy as np
import pandas as pd
import csv
import re

letters = ('I', 'E', 'N', 'S', 'T', 'F', 'P', 'J')
df = pd.read_csv('mbti_clean.csv')
test = collections.defaultdict(list) 

for row in df.iterrows():
	posts = row[1]['posts'].split('|||')
	test[row[1]['type']].append(posts)

### write csv files for every every letter class and train vs. test class (16 total)
for letter in letters: 
	PATH = 'test_' + letter + '.csv'
	with open(PATH, 'w') as f:
		writer = csv.writer(f)
		for key in test.keys():
			if letter in key:
				for hundred_posts in test[key]:
					row = [post for post in hundred_posts if ('http' not in post) and (post != '') and (post != None) and (re.search("[a-zA-Z]", post))]
					if len(row) > 10: 
						writer.writerow(row)



			
