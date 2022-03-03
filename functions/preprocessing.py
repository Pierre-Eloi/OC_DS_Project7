#! /usr/bin/env python3
# coding: utf-8

""" This module gathers all functions required to prepare data."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

class NaAttFilter(BaseEstimator, TransformerMixin):
    """Drop all attributes with a percentage of missing values higher than na_treshold.

    Parameters:
    -----------
    na_threshold: float, default=0.3
        Control the features to drop
    """
    def __init__(self, na_threshold=.3):
        self.na_threshold = na_threshold
    def fit(self, X, y=None):
        if self.na_threshold > 1:
            self.na_threshold /= 100
        self.na_perc = X.isna().sum() / X.shape[0]
        self.mask_ = self.na_perc > self.na_threshold
        return self
    def transform(self, X):
        return X.loc[:, ~self.mask_]

class SparseCleaner(BaseEstimator, TransformerMixin):
    """Fix anomalies for the following features:
    - DEF_30_CNT_SOCIAL_CIRCLE
    - DEF_60_CNT_SOCIAL_CIRCLE
    - AMT_REQ_CREDIT_BUREAU_QRT
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return (X.assign(DEF_30_CNT_SOCIAL_CIRCLE=lambda x: \
                     x['DEF_30_CNT_SOCIAL_CIRCLE'].where(x['DEF_30_CNT_SOCIAL_CIRCLE']<11,
                                                           np.nan))
                 .assign(DEF_60_CNT_SOCIAL_CIRCLE=lambda x: \
                     x['DEF_60_CNT_SOCIAL_CIRCLE'].where(x['DEF_60_CNT_SOCIAL_CIRCLE']<11,
                                                         np.nan))
                 .assign(AMT_REQ_CREDIT_BUREAU_QRT=lambda x: \
                     x['AMT_REQ_CREDIT_BUREAU_QRT'].where(x['AMT_REQ_CREDIT_BUREAU_QRT']<11,
                                                          np.nan)))

class DenseCleaner(BaseEstimator, TransformerMixin):
    """Function to fix anomalies for the following features:
    - DAYS_EMPLOYED
    - AMT_INCOME_TOTAL
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return (X.assign(DAYS_EMPLOYED_ANOM=lambda x: x['DAYS_EMPLOYED'] > 36500)
                 .assign(DAYS_EMPLOYED_ANOM=lambda x: x['DAYS_EMPLOYED_ANOM'].astype(int))
                 .assign(DAYS_EMPLOYED_ANOM=lambda x: x['DAYS_EMPLOYED_ANOM'].fillna(0))
                 .assign(DAYS_EMPLOYED=lambda x: \
                     x['DAYS_EMPLOYED'].where(x['DAYS_EMPLOYED']<36500, np.nan))
                 .assign(AMT_INCOME_TOTAL=lambda x: \
                     x['AMT_INCOME_TOTAL'].where(x['AMT_INCOME_TOTAL']<100_000_000,
                                                 np.nan)))

# Add domain knowledge features
class DomainAdder(BaseEstimator, TransformerMixin):
    """Add five domain Knowledge attributes:
    - DAYS_EMPLOYED_PERC: the percentage of the days employed relative to the client's age
    - CREDIT_INCOME_PERC: the percentage of the credit amount relative to a client's income
    - INCOME_PER_PERSON: the client's income relative to the size of the client's family
    - ANNUITY_INCOME_PERC: the percentage of the loan annuity relative to a client's income
    - CREDIT_TERM: the length of the payment in years
    Parameters:
    -----------
    add_domain_features: bool, default=True
        If False, no domain features are added
    """
    def __init__(self, add_domain=True):
        self.add_domain = add_domain

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.add_domain:
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
class SkewCleaner(BaseEstimator, TransformerMixin):
    """Transform data with an abs(skewness) > threshold
    For data with negative skewness, data are reflected first.
    Parameters:
    -----------
    log: bool, default False
        If True use the log transformation otherwise use the square root
    threshold: float, default=1.0
        Control the features to transform
    """
    def __init__(self, log=True, threshold=1.):
        self.log = log
        self.threshold = threshold

    def fit(self, X, y=None):
        # Keep only features with at least 100 distinct elements
        mask = list(X.loc[:, X.nunique() > 100])
        # get features to transform
        self.neg_skew_att_ = list(X[mask].loc[:, X[mask].skew() < -self.threshold])
        self.pos_skew_att_ = list(X[mask].loc[:, X[mask].skew() > self.threshold])
        # get max for negative skewness features
        self.max = np.array(X[self.neg_skew_att_].max())
        return self

    def transform(self, X):
        if self.log:
            dic_neg = dict(zip(self.neg_skew_att_,
                np.log(1 + self.max - X[self.neg_skew_att_].values).T))
            dic_pos = dict(zip(self.pos_skew_att_,
                np.log(1 + X[self.pos_skew_att_].values).T))
        else:
            dic_neg = dict(zip(self.neg_skew_att_,
                np.sqrt(1 + self.max - X[self.neg_skew_att_].values).T))
            dic_pos = dict(zip(self.pos_skew_att_,
                np.sqrt(1 + X[self.pos_skew_att_].values).T))
        return (X.assign(**dic_neg)
                 .assign(**dic_pos))

    def get_feature_names(self):
        if self.log:
            neg_att_names = [c + "_log_rev" for c in self.neg_skew_att_]
            pos_att_names = [c + "_log" for c in self.pos_skew_att_]
        else:
            neg_att_names = [c + "_sqrt_rev" for c in self.neg_skew_att_]
            pos_att_names = [c + "_sqrt" for c in self.pos_skew_att_]
        return dict(zip(self.neg_skew_att_ + self.pos_skew_att_,
                        neg_att_names + pos_att_names))

# Feature Selector
class TopFeatureSelector(BaseEstimator, TransformerMixin):
    """Select features based on the mask entered in parameter
    Parameters:
    -----------
    feature_mask: : array-like
        Mask to filter features
    """
    def __init__(self, feature_mask):
        self.feature_mask = feature_mask
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[:, self.feature_mask]
