#! /usr/bin/env python3
# coding: utf-8

"""Script to get the final model.
The latter will be saved with pickle"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from functions.preprocessing import NaAttFilter
from functions.preprocessing import SparseCleaner, DenseCleaner
from functions.preprocessing import DomainAdder
from functions.preprocessing import SkewCleaner
from functions.preprocessing import TopFeatureSelector
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import imblearn.pipeline as imbpipe
import shap
import pickle

app_train = pd.read_csv('data/application_train.csv')
X = app_train.drop(['SK_ID_CURR', 'TARGET'], axis=1)
y = app_train['TARGET']
# Classifier parameter found by grid search
classifier = LogisticRegression(C=0.01, random_state=42, solver='liblinear')
cv = StratifiedKFold(5, shuffle=True, random_state=42)
# Get the categorical attributes
cat_att = X.select_dtypes('object').columns
# Get the numerical attributes
num_att = X.select_dtypes(['number']).columns
ord_att = X[num_att].loc[:, X[num_att].nunique()<6].columns
sparse_att = np.array([c for c in num_att
                       if c not in ord_att
                       and (X[c]==0).sum() > 0.5*len(X)])
dense_att = np.array([c for c in num_att
                      if c not in ord_att
                      and c not in sparse_att])
# Create a pipeline for data preparation
# Pipeline parameters found by grid search
cat_pipeline = Pipeline([
    ('filter', NaAttFilter(na_threshold=0.4)),
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='if_binary')),
    ])
ord_pipeline = Pipeline([
    ('filter', NaAttFilter(na_threshold=0.4)),
    ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
sparse_pipeline = Pipeline([
    ('filter', NaAttFilter(na_threshold=0.4)),
    ('cleaner', SparseCleaner()),
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', MaxAbsScaler())
    ])
dense_pipeline = Pipeline([
    ('filter', NaAttFilter(na_threshold=0.4)),
    ('cleaner', DenseCleaner()),
    ('domain_adder', DomainAdder(add_domain=True)),
    ('skew_transformer', SkewCleaner(log=False, threshold=1.6)),
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
    ])
prep_pipeline = ColumnTransformer([
    ('cat', cat_pipeline, cat_att),
    ('ord', ord_pipeline, ord_att),
    ('sparse', sparse_pipeline, sparse_att),
    ('dense', dense_pipeline, dense_att),
    ])
transformer = Pipeline([
    ('preprocessing', prep_pipeline),
    ('feature_selection', TopFeatureSelector(feature_mask=None))
    ])
# Create a sampling pipeline to balance the dataset
# Pipeline parameters found by grid search
sampling_pipeline = imbpipe.Pipeline([
    ('over', SMOTE(random_state=42, sampling_strategy=0.3)),
    ('under', RandomUnderSampler(random_state=42, sampling_strategy=0.6))
    ])

def get_feature_mask(X, y):
    """ Feature selection with recurcive feature elimination.
    -----------
    Parameters:
    X: array-like of shape (n_samples, n_features)
        The training input samples
    y: array-like of shape (n_samples)
        The target values
    -----------
    Return:
        The mask of selected features
    """
    clf = SGDClassifier(loss='hinge', random_state=42, average=True)
    selector = RFE(clf, n_features_to_select=80, step=5)
    selector.fit(X, y)
    return selector.support_

def get_feature_names(X, feature_mask):
    """ Function to get the list of feature names after data preparation
    -----------
    Parameters:
    X: array-like of shape (n_samples, n_features)
        The training input samples
    feature_mask: array_like
        The mask of selected features
    -----------
    Return:
        The mask of selected features
    """
    # Get the name of the transformed categorical features
    cat_pipeline.fit(X[cat_att]) # required to get attributes, ColumnTransformer bug?
    cat_mask = prep_pipeline.get_params()['cat__filter'].mask_
    encoder = prep_pipeline.get_params()['cat__encoder']
    cat_att_tr = encoder.get_feature_names_out(cat_att[~cat_mask])
    # Get the name of the transformed ordinal features
    ord_pipeline.fit(X[ord_att])
    ord_mask = prep_pipeline.get_params()['ord__filter'].mask_
    ord_att_tr = ord_att[~ord_mask]
    # Get the name of the transformed sparse features
    sparse_pipeline.fit(X[sparse_att])
    sparse_mask = prep_pipeline.get_params()['sparse__filter'].mask_
    sparse_att_tr = sparse_att[~sparse_mask]
    # Get the name of the transformed dense features
    dense_pipeline.fit(X[dense_att])
    dense_mask = prep_pipeline.get_params()['dense__filter'].mask_
    dense_att_tr = dense_att[~dense_mask]
    domain_att = ['DAYS_EMPLOYED_PERC',
                  'CREDIT_INCOME_PERC',
                  'INCOME_PER_PERSON',
                  'ANNUITY_INCOME_PERC',
                  'CREDIT_TERM']
    extra_att = np.concatenate((['DAYS_EMPLOYED_ANOM'], domain_att), axis=None)
    skew_transformer = prep_pipeline.get_params()['dense__skew_transformer']
    dense_att_tr = (pd.Series(dense_att_tr)
                      .append(pd.Series(extra_att))
                      .replace(skew_transformer.get_feature_names()))
    # Concatenate all attributes
    all_att = np.concatenate(
        (cat_att_tr, ord_att_tr, sparse_att_tr, dense_att_tr), axis=None
        )
    return all_att[feature_mask]

def score_predictor(X):
    """Predict a score with a 90% Recall.
    -----------
    Parameters:
    X: array-like
        The data to predict
    -----------
    Return:
        predicted score
    """
    return classifier.predict_proba(X)[:, 1] > 0.2

def main():
    X_pr = prep_pipeline.fit_transform(X)
    X_sampled, y_sampled = sampling_pipeline.fit_resample(X_pr, y)
    # Get the list of features
    feature_mask = get_feature_mask(X_sampled, y_sampled)
    features_selected = get_feature_names(X, feature_mask)
    # Train the Classifier
    transformer['feature_selection'].feature_mask=feature_mask
    X_tr = transformer.fit_transform(X)
    X_sampled, y_sampled = sampling_pipeline.fit_resample(X_tr, y)
    classifier.fit(X_sampled, y_sampled)
    # Get the SHAP explainer
    means = np.mean(transformer.transform(X), axis=0).reshape(1, -1)
    explainer = shap.KernelExplainer(score_predictor, means)
    # Save the model with Pickle
    model = {'transformer': transformer,
             'features_selected': features_selected,
             'classifier': classifier}
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    # Save the explainer with Pickle
    with open('explainer.pkl', 'wb') as file:
        pickle.dump(explainer, file)

if __name__ == "__main__":
    main()
