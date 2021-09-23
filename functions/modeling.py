#! /usr/bin/env python3
# coding: utf-8

""" This module gathers all functions required to train a classifier."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer

cv = StratifiedKFold(5, shuffle=True, random_state=42)

def get_models():
    """Get a list of the most common classification models to evaluate:
    - Ridge classifier
    - Logistic Regression
    - SVM, linear kernel
    - K Neighbors Classifier
    - Gaussian Naive Bayes
    - Random Forest Classifier
    - Gradient Boosting Classifier
    - XGBoost Classifier
    - CatBoost Classifier
    - LightGBM
    -----------
    Return:
        dict
    """
    models = {}
    models['Ridge Classifier'] = RidgeClassifier()
    models['Logistic Regression'] = SGDClassifier(loss='log', random_state=42)
    models['SVM - Linear kernel'] = SGDClassifier(loss='hinge', random_state=42)
    models['K Neighbors Classifier'] = KNeighborsClassifier()
    models['Gaussian Naive Bayes'] = GaussianNB()
    models['Decision Tree Classifier'] = DecisionTreeClassifier(random_state=42)
    models['Random Forest Classifier'] = RandomForestClassifier(random_state=42)
    models['Gradient Boosting Classifier'] = GradientBoostingClassifier(random_state=42)
    models['XGBoost Classifier'] = XGBClassifier(random_state=42)
    models['CatBoost Classifier'] = CatBoostClassifier()
    models['LightGBM Classifier'] = LGBMClassifier()
    return models

def pr_auc(y_true, probas_pred):
    """ Compute Area Under the Precision-Recall Curve.
    -----------
    Parameters:
    y_true: ndarray of shape (n_samples,)
        True binary labels.
    probas_pred: ndarray of shape (n_samples,)
        Estimated probabilities or output of a decision function.
    -----------
    Return:
        auc : float
    """
    p, r, thresholds = precision_recall_curve(y_true, probas_pred)
    return auc(r, p)

def compare_models(X, y, sort='Accuracy', auc='roc'):
    """ Fonction to compare all models listed in the get_models function.
    In addition of the fit_time, the following metrics are displayed
    - Accuracy
    - Precision
    - Recall
    - F1
    - AUC
    -----------
    Parameters:
    X: array-like
        The data to fit
    y: array-like
        The target
    sort: str
        the metrics to be used to sort the models
        The possible options are 'fit time (s)', 'Accuracy', 'Precision',
        'Recall', 'F1 Score', 'ROC AUC' or 'PR AUC'
    auc: str, default='roc'
        The curve on which to apply the auc function.
        The possible options are 'roc' or 'pr'.
        'pr' is for Precision versus Recall curve.
    -----------
    Return:
        DataFrame
    """
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score),
               'f1': make_scorer(f1_score)}

    columns = ['fit time (s)', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
    scores = []
    if auc == 'pr':
        scoring['pr_auc'] = make_scorer(pr_auc, needs_threshold=True)
        columns.append('PR AUC')
    else:
        scoring['roc_auc'] = make_scorer(roc_auc_score, needs_threshold=True)
        columns.append('ROC AUC')
    # get models
    models = get_models()
    # get scores for each model
    for m in models:
        print(m)
        m_scores = cross_validate(models[m], X, y,
                                  scoring=scoring, cv=cv, n_jobs=-1)
        m_scores = [m_scores[key].mean() for key in m_scores
                    if key != 'score_time']
        scores.append(m_scores)
    # Return a DataFrame with all scores
    return (pd.DataFrame(scores, columns=columns, index=models.keys())
              .sort_values(sort, ascending=False))
