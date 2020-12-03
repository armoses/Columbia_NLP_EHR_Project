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
from preprocessing import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sb

#%% Load and preprocess data (with and without resampling)

raw_data = pd.read_csv('mtsamples.csv', index_col=0) 
train_labels, test_labels, vec_dict, tfidf_dict = preprocess(raw_data, resample=False)
rs_train_labels, rs_test_labels, rs_vec_dict, rs_tfidf_dict = preprocess(raw_data, 
                                                                         resample=True,
                                                                         max_features=6000,
                                                                         ngram_range=(1,3))

### Find best number of features to use for our fave models ####

# from sklearn.naive_bayes import ComplementNB
# cnb = ComplementNB()

# feat_range = [3000, 4000, 5000, 6000, 7000, 8000]
# scores = []
# for i in feat_range:
#     print(f'i = {i}')
#     rs_train_labels, rs_test_labels, rs_vec_dict, rs_tfidf_dict = preprocess(raw_data, 
#                                                                          resample=True,
#                                                                          max_features=i,
#                                                                          ngram_range=(1,3))
#     cnb.fit(rs_tfidf_dict['trscrp_train_data_tfidf'], rs_train_labels)
#     this_score = cnb.score(rs_tfidf_dict['trscrp_test_data_tfidf'], rs_test_labels)
#     scores.append(this_score)
#     print(f'score = {this_score}')
# # scores = [0.56, 0.59, 0.59, 0.60, 0.59, 0.58]


# from sklearn.neural_network import MLPClassifier
# mlp = MLPClassifier(random_state=3, max_iter=1)

# feat_range = [100, 300, 500, 700, 900]
# scores = []
# for i in feat_range:
#     print(f'i = {i}')
#     rs_train_labels, rs_test_labels, rs_vec_dict, rs_tfidf_dict = preprocess(raw_data, 
#                                                                           resample=True,
#                                                                           max_features=i,
#                                                                           ngram_range=(1,3))
#     mlp.fit(rs_tfidf_dict['trscrp_train_data_tfidf'], rs_train_labels)
#     this_score = mlp.score(rs_tfidf_dict['trscrp_test_data_tfidf'], rs_test_labels)
#     scores.append(this_score)
#     print(f'score = {this_score}')
# # scores = [0.38, 0.46, 0.52, 0.55, 0.53]


#%% Complement Naive Bayes

from sklearn.naive_bayes import ComplementNB
cnb = ComplementNB()

cnb.fit(rs_tfidf_dict['trscrp_train_data_tfidf'], rs_train_labels)
cnb_pred = cnb.predict(rs_tfidf_dict['trscrp_test_data_tfidf'])
class_rep_cnb_trscrp_rs = classification_report(cnb_pred, rs_test_labels, zero_division=0) 
# 0.77 accuracy when resample train and test together, 0.55 accuracy when resampled separately
# 0.6 when you add params max_features=6000 and ngram_range=(1,3) to tfidf vectorizer

cnb.fit(tfidf_dict['trscrp_train_data_tfidf'], train_labels)
cnb_pred = cnb.predict(tfidf_dict['trscrp_test_data_tfidf'])
class_rep_cnb_trscrp = classification_report(cnb_pred, test_labels, zero_division=0) 
# 0.33 accuracy without resampling


#%% Dimensionality Reduction (PCA)

from sklearn.decomposition import PCA
pca = PCA(n_components=.95)

# Not resampled
trscrp_vec_train = pd.DataFrame(pca.fit_transform(vec_dict['trscrp_train_data_vec']))
trscrp_vec_test = pd.DataFrame(pca.transform(vec_dict['trscrp_test_data_vec']))
trscrp_tfidf_train = pd.DataFrame(pca.fit_transform(rs_tfidf_dict['trscrp_train_data_tfidf']))
trscrp_tfidf_test = pd.DataFrame(pca.transform(rs_tfidf_dict['trscrp_test_data_tfidf']))

kwords_vec_train = pd.DataFrame(pca.fit_transform(vec_dict['kwords_train_data_vec']))
kwords_vec_test = pd.DataFrame(pca.transform(vec_dict['kwords_test_data_vec']))
kwords_tfidf_train = pd.DataFrame(pca.fit_transform(tfidf_dict['kwords_train_data_tfidf']))
kwords_tfidf_test = pd.DataFrame(pca.transform(tfidf_dict['kwords_test_data_tfidf']))

#%% Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=1)

### use grid search to find best params for this one setup... (tfidf with PCA) ###
# from sklearn.model_selection import GridSearchCV
# params =  {'n_estimators':[50, 100, 200],
#             'max_depth' : [2,4,8]}
# rfc_grid = GridSearchCV(estimator=rfc, param_grid=params, cv= 3).fit(trscrp_tfidf_train, train_labels)
# best params: {'max_depth': 4, 'n_estimators': 50}

rfc = RandomForestClassifier(random_state=1, max_depth=4, n_estimators=50).fit(vec_dict['trscrp_train_data_vec'], train_labels)
rfc_pred = rfc.predict(vec_dict['trscrp_test_data_vec'])
class_rep_rfc_trscrp = classification_report(rfc_pred, test_labels, zero_division=0)
# rfc.score(vec_dict['trscrp_test_data_vec'], test_labels)

rfc = RandomForestClassifier().fit(vec_dict['kwords_train_data_vec'], train_labels)
rfc_pred = rfc.predict(vec_dict['kwords_test_data_vec'])
class_rep_rfc_kwords = classification_report(rfc_pred, test_labels, zero_division=0)

h = confusion_matrix(cnb.predict(tfidf_dict['trscrp_test_data_tfidf']), test_labels, normalize='true')
sb.heatmap(h)

rfc = RandomForestClassifier(random_state=1).fit(rs_vec_dict['trscrp_train_data_vec'], rs_train_labels)
rs_rfc_pred = rfc.predict(rs_vec_dict['trscrp_test_data_vec'])
class_rep_rs_rfc_trscrp = classification_report(rs_rfc_pred, rs_test_labels, zero_division=0)


#%% K-Nearest Neighbors (plots are saved)
from sklearn.neighbors import KNeighborsClassifier

#Setup arrays to store training and test accuracies
neighbors = np.arange(1,9)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

from tqdm import tqdm
for i,k in tqdm(enumerate(neighbors)): #train with tfidf
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    
    #Fit the model
    # knn.fit(tfidf_dict["trscrp_train_data_tfidf"], train_labels)
    knn.fit(kwords_tfidf_train, train_labels)
    
    #Compute accuracy on the training set
    # train_accuracy[i] = knn.score(tfidf_dict["trscrp_train_data_tfidf"], train_labels)
    train_accuracy[i] = knn.score(kwords_tfidf_train, train_labels)
    
    #Compute accuracy on the test set
    # test_accuracy[i] = knn.score(tfidf_dict["trscrp_test_data_tfidf"], test_labels) 
    test_accuracy[i] = knn.score(kwords_tfidf_test, test_labels) 

#Generate plot
plt.title('k-NN Varying number of neighbors: kwords_tfidf')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()

# train with kwords_vec
neighbors = np.arange(1,9)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
for i,k in tqdm(enumerate(neighbors)):
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    
    #Fit the model
    knn.fit(kwords_vec_train, train_labels)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(kwords_vec_train, train_labels)
    
    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(kwords_vec_test, test_labels) 

#Generate plot
plt.title('k-NN Varying number of neighbors: kwords_vec')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()

# train with rs_data
neighbors = np.arange(1,9)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
for i,k in tqdm(enumerate(neighbors)):
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    
    #Fit the model
    knn.fit(rs_tfidf_dict['trscrp_train_data_tfidf'], rs_train_labels)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(rs_tfidf_dict['trscrp_train_data_tfidf'], rs_train_labels)
    
    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(rs_tfidf_dict['trscrp_test_data_tfidf'], rs_test_labels) 

#Generate plot
plt.title('k-NN Varying number of neighbors: rs_train data')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()

#%% Neural Network

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state=3, max_iter=300).fit(trscrp_vec_train, train_labels)
mlp_pred = mlp.predict(trscrp_vec_test)
class_rep_mlp_trscrp = classification_report(mlp_pred, test_labels, zero_division=0)

# with resampled data
mlp = MLPClassifier(random_state=3, max_iter=1).fit(rs_tfidf_dict['trscrp_train_data_tfidf'], rs_train_labels)
mlp_pred = mlp.predict(rs_tfidf_dict['trscrp_test_data_tfidf'])
class_rep_mlp_trscrp = classification_report(mlp_pred, rs_test_labels, zero_division=0) 
# 0.79 accuracy (2 iterations) resample train and test together
# 0.52 resampled separately, max_features=6000, ngram_range=(1,3)

#%% Experimental stuff

##### LDA
from sklearn.decomposition import LatentDirichletAllocation
n_labels = len(raw_data.medical_specialty.unique())

def run_lda(train_df, test_df, train_labels):
    print('Running LDA...')
    lda = LatentDirichletAllocation(random_state=0, n_components=n_labels)
    train_trans = lda.fit_transform(train_df)
    lda_predicted_labels = pd.DataFrame(lda.transform(test_df))
    
    print('Getting test predictions...')
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
    
    print('Finding key words for each specialty...')
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
                                                              vec_dict['trscrp_test_data_vec'],
                                                              train_labels) # 0.35 accuracy

lda_kwords_vec_test, lda_kwords_vec_specialty_words = run_lda(vec_dict['kwords_train_data_vec'], 
                                                              vec_dict['kwords_test_data_vec'])

# Evaluate
class_rep_lda_trscrp = classification_report(lda_trscrp_vec_test, rs_test_labels, zero_division=0)
class_rep_lda_kwords = classification_report(lda_kwords_vec_test, test_labels, zero_division=0)


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
    


test_binary_labels = np.array(rs_test_labels.copy())
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
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=8)
from sklearn.naive_bayes import ComplementNB
cnb = ComplementNB()


# binary_pred, all_predictions = multi_step(vec_dict['trscrp_train_data_vec'], 
#                                           vec_dict['trscrp_test_data_vec'],
#                                           train_labels,log_reg, rfc)
binary_pred, all_predictions = multi_step(rs_tfidf_dict['trscrp_train_data_tfidf'], 
                                          rs_tfidf_dict['trscrp_test_data_tfidf'],
                                          rs_train_labels,log_reg, cnb)

# Evaluate binary and full
class_rep_bin = classification_report(binary_pred, test_binary_labels, zero_division=0)
class_rep_full = classification_report(all_predictions, test_labels, zero_division=0)
