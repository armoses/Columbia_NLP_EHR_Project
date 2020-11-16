#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 22:41:12 2020

@author: aimeemoses
"""
#%% Imports

import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

#%% Load and explore data

# Load data
raw_data = pd.read_csv('mtsamples.csv', index_col=0) # 4999 x 5

# Look at column names and nulls
raw_data.columns # 'description', 'medical_specialty', 'sample_name', 'transcription','keywords'
raw_data.info() # nulls in 'transcription' and 'keywords' columns

# See how many unique entries for each column
unique_cols = []
for col in raw_data.columns:
    unique_cols.append(pd.Series(raw_data[col].unique()))
    print(f'{col}: {len(raw_data[col].unique())}')
# description: 2348
# medical_specialty: 40 # 40 possible classes
# sample_name: 2377
# transcription: 2358
# keywords: 3850

# See all possible keywords
keywords = raw_data.keywords.loc[raw_data.keywords.isna() == False].tolist()
keywords = ' '.join(keywords)
keywords = keywords.split(',')
keywords = [token.lower().strip() for token in keywords]

unique_keywords = set(keywords) # 11,387 unique keywords (more like key-phrases)

# Not all transcriptions are unique-- are there duplicate rows?
raw_data = raw_data.drop_duplicates()
# No duplicates...
sorted_data = raw_data.sort_values(by=['sample_name', 'transcription', 'medical_specialty'])
# Each transcription/description/sample_name may appear in multiple row
# for each medical specialty it is classified as (and different keywords, accordingly)
# Do we want to use each sample just once or allow multiple classifications?
# Perhaps we want to do LDA to get a probability/rating of belonging to different specialties

# Plot lengths of entries
desc_lengths = unique_cols[0].apply(lambda x: len(x.split()))
plt.hist(desc_lengths)
plt.title('lengths of descriptions')
plt.xlabel('words')
plt.ylabel('samples')
plt.show()

trans_lengths = unique_cols[3].loc[unique_cols[3].isna() == False].apply(lambda x: len(x.split()))
plt.hist(trans_lengths)
plt.title('lengths of transcriptions')
plt.xlabel('words')
plt.ylabel('samples')
plt.show()

key_lengths = sorted([len(key) for key in keywords])[:-253] # 253 key-phrases are >543 words
plt.hist(key_lengths)
plt.title('lengths of key-phrases')
plt.xlabel('words')
plt.ylabel('samples')
plt.show()

name_lengths = unique_cols[2].apply(lambda x: len(x.split()))
plt.hist(name_lengths, bins=8)
plt.title('lengths of sample_names')
plt.xlabel('words')
plt.ylabel('samples')
plt.show()

# Plot distribution of labels
plt.figure(figsize=(12,8))
labels, label_counts = np.unique(raw_data['medical_specialty'], return_counts=True)
labels = [val for _,val in sorted(zip(label_counts,labels))]
label_counts = sorted(label_counts)
plt.bar(labels, height=label_counts)
plt.title('label counts')
plt.xlabel('labels')
plt.ylabel('counts')
plt.xticks(rotation=90)
plt.show()


#%% Pre-process

# What data do we want to consider (description vs transcription vs sample name vs keywords)?
# Perhaps we try training on each of them individually as well as joined together, 
# and see what gets the best performance...?

# Train-test split
from sklearn.model_selection import train_test_split
np.random.seed(5) # Scikit Learn does not have its own global random state but uses the numpy random state instead.
train, test = train_test_split(raw_data, test_size=0.2)  # split to 8:2 training:test

# Get dataframes with respective column(s) for each of the options above
dscrp_data = raw_data["description"]
med_special_data = raw_data["medical_speciality"]
trscrp_data = raw_data["transcription"]
samp_name_data = raw_data["sample_name"]
kwords_data = raw_data["keywords"]


# Remove stop words
no_stopw_data = raw_data
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
stop_words = stopwords.words('english')

for i in no_stopw_data.columns: # get cols
    no_stopw_data[i] = no_stopw_data[i].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in (stop_words)])) # I think the code works, but looks like two dataset are not different. 

# Stem
stem_data = no_stopw_data

from nltk.stem import PorterStemmer
ps = PorterStemmer()
for i in stem_data.columns: # get cols
    stem_data[i] = stem_data[i].apply(lambda x: " ".join([ps.stem(word) for word in word_tokenize(x)]))

        
# Count vectorizer


# Tf-idf









