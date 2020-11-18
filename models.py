#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 14:43:43 2020

@author: aimeemoses
"""
#%% Imports

import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import pickle
from preprocessing import preprocess

#%% Load and preprocess data

raw_data = pd.read_csv('mtsamples.csv', index_col=0) 
train_data, test_data, train_labels, test_labels, vec_dict, tfidf_dict = preprocess(raw_data)

#%% Classic supervised models



#%% Experimental stuff

# LDA
from sklearn.decomposition import LatentDirichletAllocation
n_labels = len(raw_data.medical_specialty.unique())

def run_lda(train_df, test_df):
    lda = LatentDirichletAllocation(random_state=0, n_components=n_labels)
    train_trans = lda.fit_transform(train_df)
    lda_predicted_labels = pd.DataFrame(lda.transform(test_df))
    
    model_comps = pd.DataFrame(lda.components_)
    model_comps = model_comps.div(model_comps.sum(axis=1), axis=0)
    model_comps.columns = train_df.columns
    
    component_labels = {}      
    train_comp_labels = pd.DataFrame(train_trans).idxmax(axis=1)
    for component in range(40):
        this_comp_labels = train_labels[train_comp_labels == component]
        if len(this_comp_labels) > 0:
            component_labels[component] = this_comp_labels.mode()[0]
        else:
            component_labels[component] = 'no_label'
    lda_predicted_labels.columns = component_labels.values()
    lda_predicted_labels = lda_predicted_labels.idxmax(axis=1)
    
    label_components = {}
    for label in train_labels.unique():
        this_label_df = pd.DataFrame(train_trans[train_labels == label])
        label_components[label] = this_label_df.sum().idxmax()
    specialty_words = {}
    for label in label_components.keys():
        top_cols = model_comps.T.nlargest(n=5, columns=model_comps.T.columns[label_components[label]])
        specialty_words[label] = list(top_cols.index)

    return lda_predicted_labels, specialty_words

lda_trscrp_vec_test, lda_trscrp_vec_specialty_words = run_lda(vec_dict['trscrp_train_data_vec'], 
                                                              vec_dict['trscrp_test_data_vec'])

# Evaluate
from sklearn.metrics import confusion_matrix
import seaborn as sb
h = confusion_matrix(lda_trscrp_vec_test, test_labels, normalize='true')
sb.heatmap(h)

from sklearn.metrics import classification_report
class_rep = classification_report(lda_trscrp_vec_test, test_labels, zero_division=0)



# Binary then multi-class models


