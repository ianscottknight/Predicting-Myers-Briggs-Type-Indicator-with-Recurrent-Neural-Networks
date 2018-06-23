#!/usr/bin/env python

import random
import collections
import numpy as np
import pandas as pd
import csv
import re

df = pd.read_csv('mbti_unclean.csv')
DIMENSIONS = ('IE', 'NS', 'TF', 'PJ')
counts = collections.defaultdict(int)

for dimension in DIMENSIONS: 
	letter_1, letter_2 = dimension
	for mbti in df['type']:
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
	list1 = []
	list2 = []
	i = 0
	for row in df.iterrows():	
		if i == limit: break
		if letter_1 in row[1]['type']:
			list1.append(row[1]['posts'].split('|||'))
			i += 1
	i = 0
	for row in df.iterrows():	
		if i == limit: break
		if letter_2 in row[1]['type']:
			list2.append(row[1]['posts'].split('|||'))	
			i += 1
	with open('train_' + letter_1 + '.csv', 'w') as f:
		writer = csv.writer(f)
		for hundred_posts in list1:
			row = [post for post in hundred_posts if ('http' not in post) and (post != '') and (post != None) and (re.search("[a-zA-Z]", post))]
			writer.writerow(row)
	with open('train_' + letter_2 + '.csv', 'w') as f:
		writer = csv.writer(f)
		for hundred_posts in list2:
			row = [post for post in hundred_posts if ('http' not in post) and (post != '') and (post != None) and (re.search("[a-zA-Z]", post))]
			writer.writerow(row)



	