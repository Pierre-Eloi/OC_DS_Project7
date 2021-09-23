#! /usr/bin/env python3
# coding: utf-8

""" This module gathers all functions required to prepare data."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

def drop_na_att(X, na_threshold=.3):
    """Drop all attributes with a percentage of missing values higher than na_treshold.
    -----------
    Parameters:
    X: DataFrame
        The input samples
    na_threshold: float, default=0.3
        Control the features to drop
    """
    if na_threshold > 1:
        na_threshold /= 100
    mask = X.isna().sum()/len(X) > na_threshold
    att_to_del = list(X.loc[:, mask])
    return X.drop(att_to_del, axis=1)

def impute_cat_att(X, values=None):
    """Impute missing values in categorical attribute with the most frequent category
    -----------
    Parameters:
    X: DataFrame
        The input samples
    values: Series, default=None
        Values to use to fill holes.
    """
    if values is None:
        values = drop_na_att(X).value_counts().index[0]
    cat_att = list(X.select_dtypes('object'))
    cat_val = dict(zip(cat_att, values))
    return X[cat_att].fillna(cat_val)

def fix_sparse_anomalies(X):
    """Function to fix anomalies for the following features:
    - DEF_30_CNT_SOCIAL_CIRCLE
    - DEF_60_CNT_SOCIAL_CIRCLE
    - AMT_REQ_CREDIT_BUREAU_QRT
    -----------
    Parameters:
    X: DataFrame
        The input samples
    """
    return (X.assign(DEF_30_CNT_SOCIAL_CIRCLE=lambda x: \
                     x['DEF_30_CNT_SOCIAL_CIRCLE'].where(x['DEF_30_CNT_SOCIAL_CIRCLE']<11,
                                                           np.nan))
             .assign(DEF_60_CNT_SOCIAL_CIRCLE=lambda x: \
                     x['DEF_60_CNT_SOCIAL_CIRCLE'].where(x['DEF_60_CNT_SOCIAL_CIRCLE']<11,
                                                         np.nan))
             .assign(AMT_REQ_CREDIT_BUREAU_QRT=lambda x: \
                     x['AMT_REQ_CREDIT_BUREAU_QRT'].where(x['AMT_REQ_CREDIT_BUREAU_QRT']<11,
                                                          np.nan)))

def fix_dense_anomalies(X):
    """Function to fix anomalies for the following features:
    - DAYS_EMPLOYED
    - AMT_INCOME_TOTAL
    -----------
    Parameters:
    X: DataFrame
        The input samples
    """
    return (X.assign(DAYS_EMPLOYED_ANOM=lambda x: x['DAYS_EMPLOYED'] > 36500)
             .assign(DAYS_EMPLOYED_ANOM=lambda x: x['DAYS_EMPLOYED_ANOM'].astype(int))
             .assign(DAYS_EMPLOYED_ANOM=lambda x: x['DAYS_EMPLOYED_ANOM'].fillna(0))
             .assign(DAYS_EMPLOYED=lambda x: \
                     x['DAYS_EMPLOYED'].where(x['DAYS_EMPLOYED']<=36500, np.nan))
             .assign(AMT_INCOME_TOTAL=lambda x: \
                     x['AMT_INCOME_TOTAL'].where(x['AMT_INCOME_TOTAL']<100_000_000,
                                                 np.nan)))

# Add polynomial features
def add_polynomial_att(X, names, degree=2, add_poly_features=True):
    """Generate polynomial and interaction features.
    Only the two most TARGET correlated features are taken into account:
    - 'EXT_SOURCE_2'
    - 'EXT_SOURCE_3'
    -----------
    Parameters:
    X: array_like
        The input samples
    names: list-like
        A list of columns names
    degree: int, default=3
        The degree of the polynomial features
    add_poly_features: bool, default=True
        If False, no polynomial features are added
    """
    # Tansform the array to a DataFrame
    df = pd.DataFrame(X, columns=names)
    if add_poly_features:
        poly_att = ['EXT_SOURCE_2', 'EXT_SOURCE_3']
        n = len(poly_att)
        # create the polynomial object with specified degree
        poly_transformer = PolynomialFeatures(degree=degree, include_bias=False)
        # Transform the features
        X_tr = poly_transformer.fit_transform(df[poly_att])[:, n:]
        poly_df = pd.DataFrame(X_tr,
                               columns=poly_transformer.get_feature_names(poly_att)[n:])
        return pd.concat([df, poly_df], axis=1)
    else:
        return df

# Add domain knowledge features
def add_domain_att(X, add_domain_features=True):
    """Function to add five domain Knowledge attributes:
    - DAYS_EMPLOYED_PERC: the percentage of the days employed relative to the client's age
    - CREDIT_INCOME_PERC: the percentage of the credit amount relative to a client's income
    - INCOME_PER_PERSON: the client's income relative to the size of the client's family
    - ANNUITY_INCOME_PERC: the percentage of the loan annuity relative to a client's income
    - CREDIT_TERM: the length of the payment in years
    -----------
    Parameters:
    X: DataFrame
        The input samples
    add_domain_features: bool, default=True
        If False, no domain features are added
    """
    if add_domain_features:
        return (X.assign(DAYS_EMPLOYED_PERC=lambda x:
                         x['DAYS_EMPLOYED'] / x['DAYS_BIRTH'])
                 .assign(CREDIT_INCOME_PERC=lambda x:
                         x['AMT_CREDIT'] / x['AMT_INCOME_TOTAL'])
                 .assign(INCOME_PER_PERSON=lambda x:
                         x['AMT_INCOME_TOTAL'] / x['CNT_FAM_MEMBERS'])
                 .assign(ANNUITY_INCOME_PERC=lambda x:
                         x['AMT_ANNUITY'] / x['AMT_INCOME_TOTAL'])
                 .assign(CREDIT_TERM=lambda x:
                         x['AMT_ANNUITY'] / x['AMT_CREDIT']))
    else:
        return X

# Add Basic transformations
def tr_skew_att(X, log=True, threshold=1.):
    """Transform data with an abs(skewness) > threshold
    For data with negative skewness, data are reflected first.
    -----------
    Parameters:
    X: array-like
        The input samples
    log: bool, default False
        If True use the log transformation otherwise use the square root
    threshold: float, default=1.0
        Control the features to transform
    """
    #X = pd.DataFrame(X)
    #X.columns = X.columns.astype(str)
    # Keep only features with at least 100 distinct elements
    mask = list(X.loc[:, X.nunique() > 100])
    # get features to transform
    neg_skew_att = list(X[mask].skew().index[X[mask].skew() < -threshold])
    pos_skew_att = list(X[mask].skew().index[X[mask].skew() > threshold])
    if log:
        dic_neg = dict(zip(neg_skew_att,
                           [np.log(1 + X[c].max() - X[c])
                            for c in neg_skew_att]))
        dic_pos = dict(zip(pos_skew_att,
                           [np.log(1 + X[c])
                            for c in pos_skew_att]))
        return (X.assign(**dic_neg)
                 .assign(**dic_pos))
    else:
        dic_neg = dict(zip(neg_skew_att,
                           [np.sqrt(1 + X[c].max() - X[c])
                            for c in neg_skew_att]))
        dic_pos = dict(zip(pos_skew_att,
                           [np.sqrt(1+ X[c])
                            for c in pos_skew_att]))
        return (X.assign(**dic_neg)
                 .assign(**dic_pos))
