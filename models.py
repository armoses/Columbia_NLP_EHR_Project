#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 14:43:43 2020

@author: aimeemoses
"""
#%% Imports

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from preprocessing import preprocess

#%% Load and preprocess data

raw_data = pd.read_csv('mtsamples.csv', index_col=0) 
train_data, test_data, train_labels, test_labels, vec_dict, tfidf_dict = preprocess(raw_data)


#%% Dimensionality Reduction (PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components=.95)
trscrp_vec_train = pd.DataFrame(pca.fit_transform(vec_dict['trscrp_train_data_vec']))
trscrp_vec_test = pd.DataFrame(pca.transform(vec_dict['trscrp_test_data_vec']))


#%% Classic supervised models

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
rfc = RandomForestClassifier(random_state=1).fit(trscrp_vec_train, 
                                                 train_labels)
rfc_pred = rfc.predict(trscrp_vec_test)
class_rep = classification_report(rfc_pred, test_labels, zero_division=0)
# rfc.score(vec_dict['trscrp_test_data_vec'], test_labels)


#%% K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

#Setup arrays to store training and test accuracies
neighbors = np.arange(1,9)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    
    #Fit the model
    knn.fit(tfidf_dict["trscrp_train_data_tfidf"], train_labels)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(tfidf_dict["trscrp_train_data_tfidf"], train_labels)
    
    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(tfidf_dict["trscrp_test_data_tfidf"], test_labels) 

#Generate plot
plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()
    
#%% Experimental stuff

##### LDA
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


#### Binary then multi-class models
def multi_step(train_df, test_df, train_labels, binary_model, multi_model):
    train_binary_labels = train_labels.copy()
    for i in range(len(train_binary_labels)):
        if train_binary_labels[i] == 'surgeri':
            train_binary_labels[i] = 1
        else:
            train_binary_labels[i] = 0
    train_binary_labels = train_binary_labels.astype(int)
            
    bin_mod = binary_model.fit(train_df, train_binary_labels)
    binary_pred = bin_mod.predict(test_df)
    
    # Get non-surgery and run through next model
    non_surg_train = train_df[train_labels != 'surgeri']
    non_surg_train_labels = train_labels[train_labels != 'surgeri']
    non_surg_test = test_df[binary_pred == 0]
    
    mult_mod = multi_model.fit(non_surg_train, non_surg_train_labels)
    non_surg_predict = mult_mod.predict(non_surg_test)
    non_surg_predict = pd.Series(non_surg_predict)
    non_surg_predict.index = non_surg_test.index
    
    all_predictions = train_labels[:len(binary_pred)].copy()
    for i in range(len(binary_pred)):
        if i in non_surg_predict.index:
            all_predictions[i] = non_surg_predict[i]
        else:
            all_predictions[i] = 'surgeri'
    
    return binary_pred, all_predictions
    


test_binary_labels = test_labels.copy()
for i in range(len(test_binary_labels)):
    if test_binary_labels[i] == 'surgeri':
        test_binary_labels[i] = 1
    else:
        test_binary_labels[i] = 0
test_binary_labels = test_binary_labels.astype(int)

# Logistic Regression for binary
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()

# Random Forest for non-surgery
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=1)

binary_pred, all_predictions = multi_step(vec_dict['trscrp_train_data_vec'], 
                                          vec_dict['trscrp_test_data_vec'],
                                          train_labels,log_reg, rfc)
# binary_pred, all_predictions = multi_step(trscrp_vec_train, 
#                                           trscrp_vec_test,
#                                           train_labels,log_reg, rfc)

# Evaluate binary and full
class_rep_bin = classification_report(binary_pred, test_binary_labels, zero_division=0)
class_rep_full = classification_report(all_predictions, test_labels, zero_division=0)
