#! /usr/bin/env python3
# coding: utf-8

import streamlit as st
import streamlit.components.v1 as components
import s3fs
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap

# Setting page config to wide mode
st.set_page_config(layout="wide")

st.title('Python Real Time Scoring API with Home Credit data')

# description and instructions
st.write("""A real time scoring API to predict clients' repayment abilities""")
st.write(st.session_state)

# Create connection object.
# `anon=False` means not anonymous, i.e. it uses access keys to pull data.
fs = s3fs.S3FileSystem(anon=False)
@st.experimental_memo # replace @st.cache, do refer to the doc for clarifications
def load_data(filename):
    with fs.open(filename) as f:
        return (pd.read_parquet(f)
                  .set_index('SK_ID_CURR'))

data_load_state = st.text('Loading data...')
data = load_data("oc-project7-per/datasets/application_test.pqt")
data_load_state.text("Loading data...completed")
st.dataframe(data.head())