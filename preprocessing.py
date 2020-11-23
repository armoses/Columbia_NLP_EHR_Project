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
import pickle

#%% Load and explore data

def main():
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
    plt.title('lengths of key-phrases (<500)')
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
    
    fig, ax = plt.subplots()
    counts, bins, patches = ax.hist(label_counts, bins=10)
    ax.set_xticks(bins)
    plt.title('label counts')
    plt.xlabel('counts')
    plt.ylabel('frequency (40 total)')
    plt.show()
    
     # Hospice - Palliative Care: 6
     # Allergy / Immunology: 7
     # Autopsy: 8
     # Lab Medicine - Pathology: 8
     # Speech - Language: 9
     # Diets and Nutritions: 10
     # Rheumatology: 10
     # Chiropractic: 14
     # IME-QME-Work Comp etc.: 16
     # Bariatrics: 18
     # Endocrinology: 19
     # Sleep Medicine: 20
     # Physical Medicine - Rehab: 21
     # Letters: 23
     # Cosmetic / Plastic Surgery: 27
     # Dentistry: 27
     # Dermatology: 29
     # Podiatry: 47
     # Office Notes: 51
     # Psychiatry / Psychology: 53
     # Pain Management: 62
     # Pediatrics - Neonatal: 70
     # Emergency Room Reports: 75
     # Nephrology: 81
     # Ophthalmology: 83
     # Hematology - Oncology: 90
     # Neurosurgery: 94
     # ENT - Otolaryngology: 98
     # Discharge Summary: 108
     # Urology: 158
     # Obstetrics / Gynecology: 160
     # SOAP / Chart / Progress Notes: 166
     # Neurology: 223
     # Gastroenterology: 230
     # General Medicine: 259
     # Radiology: 273
     # Orthopedic: 355
     # Cardiovascular / Pulmonary: 372
     # Consult - History and Phy.: 516
     # Surgery: 1103


#%% Pre-process

# What data do we want to consider (description vs transcription vs sample name vs keywords)?
# Perhaps we try training on each of them individually as well as joined together, 
# and see what gets the best performance...?


def preprocess(raw_data):
    
    # Clean text
    def clean_text(str_in):
        str_out = re.sub("[^A-z0-9-]+", " ", str_in.lower())
        return str_out
    
    clean_data = raw_data.copy()
    for i in clean_data.columns: # get cols
        clean_data[i].loc[clean_data[i].isna() == False] = clean_data[i][clean_data[i].isna() == False].apply(clean_text)  
        clean_data[i].replace(np.nan, '', inplace=True)
    
    
    # Remove stop words
    from nltk.corpus import stopwords 
    from nltk.tokenize import word_tokenize 
    stop_words = stopwords.words('english')
    
    def remove_stopwords(str_in): # Made this a function because the line got too long with the NaN stuff
        str_out = word_tokenize(str_in) 
        str_out = [word for word in str_out if word not in (stop_words)]
        str_out = ' '.join(str_out)
        return str_out
    
    no_stopw_data = clean_data.copy()
    for i in no_stopw_data.columns: 
        no_stopw_data[i] = no_stopw_data[i].apply(remove_stopwords)  
        # I was getting an error with NaNs
        # I think the stopword dictionary is lowercase, so our text needed to be too (and without punctuation)
    
    
    # Stem 
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    
    stem_data = no_stopw_data.copy()
    for i in stem_data.columns: 
        stem_data[i] = stem_data[i].apply(lambda x: " ".join([ps.stem(word) for word in word_tokenize(x)]))
    
    
    # Train-test split
    # Split to 8:2 training:test and mimic full distribution of labels in train and test data
    from sklearn.model_selection import train_test_split
    train_data, test_data, train_labels, test_labels = train_test_split(stem_data[['sample_name', 'description', 'transcription', 'keywords']], 
                                                                        stem_data['medical_specialty'], test_size=0.2, random_state=5, 
                                                                        stratify=stem_data['medical_specialty'])  
           
    for df in [train_data, test_data, train_labels, test_labels]:
        df.reset_index(drop=True, inplace=True)
    # Get dataframes with respective column(s) for each of the options above
    # and assign names for saving purposes
    dscrp_train_data = train_data["description"].copy()
    dscrp_train_data.name = 'dscrp_train_data'
    trscrp_train_data = train_data["transcription"].copy()
    trscrp_train_data.name = 'trscrp_train_data'
    samp_name_train_data = train_data["sample_name"].copy()
    samp_name_train_data.name = 'samp_name_train_data'
    kwords_train_data = train_data["keywords"].copy()
    kwords_train_data.name = 'kwords_train_data'
    all_train_data = dscrp_train_data +' '+ trscrp_train_data +' '+ samp_name_train_data +' '+ kwords_train_data
    all_train_data.name = 'all_train_data'
    
    dscrp_test_data = test_data["description"].copy()
    dscrp_test_data.name = 'dscrp_test_data'
    trscrp_test_data = test_data["transcription"].copy()
    trscrp_test_data.name = 'trscrp_test_data'
    samp_name_test_data = test_data["sample_name"].copy()
    samp_name_test_data.name = 'samp_name_test_data'
    kwords_test_data = test_data["keywords"].copy()
    kwords_test_data.name = 'kwords_test_data'
    all_test_data = dscrp_test_data +' '+ trscrp_test_data +' '+ samp_name_test_data +' '+ kwords_test_data
    all_test_data.name = 'all_test_data'
    
    train_sets = [all_train_data, dscrp_train_data, trscrp_train_data, samp_name_train_data, kwords_train_data]
    test_sets = [all_test_data, dscrp_test_data, trscrp_test_data, samp_name_test_data, kwords_test_data]
    
    
    # Count vectorizer and tfidf vectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    def vectorize(train_series, test_series):
        vectorizer = CountVectorizer()
        train_vec = pd.DataFrame(vectorizer.fit_transform(train_series).toarray())
        test_vec = pd.DataFrame(vectorizer.transform(test_series).toarray())
        train_vec.columns = vectorizer.get_feature_names()
        test_vec.columns = vectorizer.get_feature_names()
        return train_vec, test_vec
    
    def tfidf(train_series, test_series):
        vectorizer = TfidfVectorizer()
        train_vec = pd.DataFrame(vectorizer.fit_transform(train_series).toarray())
        test_vec = pd.DataFrame(vectorizer.transform(test_series).toarray())
        train_vec.columns = vectorizer.get_feature_names()
        test_vec.columns = vectorizer.get_feature_names()
        return train_vec, test_vec
    
    vec_dict = {}
    tfidf_dict = {}
    for train_set, test_set in zip(train_sets, test_sets):
        train_vec, test_vec = vectorize(train_set, test_set)
        vec_dict[f'{train_set.name}_vec'] = train_vec
        vec_dict[f'{test_set.name}_vec'] = test_vec
        
        train_tfidf, test_tfidf = tfidf(train_set, test_set)
        tfidf_dict[f'{train_set.name}_tfidf'] = train_tfidf
        tfidf_dict[f'{test_set.name}_tfidf'] = test_tfidf
    
    ### We now have train and test versions of all columns and each column separately
    ## independently fit and transformed by count vectorizer and by tfidf vectorizer.
    ## So we have 10 different dataframes to try models on (all the train ones)    

    return train_data, test_data, train_labels, test_labels, vec_dict, tfidf_dict

# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()



