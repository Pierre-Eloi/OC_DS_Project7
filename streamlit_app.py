#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import streamlit.components.v1 as components
import s3fs
import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap


# -------VARIABLE DECLERATION-------

# Heroku API
url = 'https://oc-p7-per.herokuapp.com'
# Create connection object.
fs = s3fs.S3FileSystem(anon=False) # `anon=False` means not anonymous, i.e. it uses access keys to pull data.
data_path = "oc-project7-per/datasets/application_test.pqt"
# read the model pickle file
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
transformer = model['transformer']
feature_names = model['features_selected']


# -------FUNCTIONS-------

@st.experimental_memo # replace @st.cache, do refer to the doc for clarifications
def load_data(file_path):
    """Function to load data in parquet format from a s3 bucket"
     -----------
    Parameters:
    filepath: string
        data in json format
    -----------
    Return:
        a DataFrame with data processed
    """
    with fs.open(file_path) as f:
        raw_data = (pd.read_parquet(f)
                      .set_index('SK_ID_CURR'))
        return pd.DataFrame(transformer.transform(raw_data),
                            columns=feature_names,
                            index=raw_data.index)

@st.experimental_memo
def get_predictions(json_data):
    """Function to get prediction using POST request.
     -----------
    Parameters:
    json_data: JSON string
        data in json format
    -----------
    Return:
        a tuple with score & probability
    """
    endpoint = '/predictions/'
    response = requests.post(f'{url}{endpoint}', json=json_data)
    if response.status_code != 200:
        return None
    else:
        return json.loads(response.content)                          

@st.experimental_memo
def get_shap_values(data, means):
    """function to get shap values using POST rsquest
     -----------
    Parameters:
    json_data: JSON string
        data in json format
    data: pandas DataFrame
        data
    means: array
        means of each feature
    -----------
    Return:
        a tuple with shap values & expected value
    """
    endpoint = '/shap/'
    #response = requests.post(f'{url}{endpoint}', json=json_data)
    response = requests.post(f'{url}{endpoint}',
                             json={'data':data.to_dict(), 'means': means.tolist()})
    if response.status_code != 200:
        
        return None
    else:
        shap_values, exp_value = json.loads(response.content)
        shap_values = np.array(shap_values).reshape(1, -1)
        return shap_values, exp_value

# Reinit the session state when the user enters a new client
def reinit_results():
    st.session_state['get_results'] = False

# Plot a force plot in streamlit API
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# Get the top features list
def top_features(shap_values, feature_names, max_display=10):
    """Sort features by importance and display the top ones.
    -----------
    Parameters:
    shap_values: array-like
        matrix of SHAP values
    feature_names: list
        Names of the features
    max_display: int
        How many top features to include
    -----------
    Return:
        list of top features
    """
    return (pd.DataFrame({'feature_names': feature_names,
                          'feature_importance': np.abs(shap_values).flatten()})
              .sort_values(by='feature_importance', ascending=False)
              .iloc[:max_display, 0]
              .to_list())


# -------DASHBOARD-------

st.set_page_config(layout="wide")

st.title('Python Real Time Scoring API with Home Credit data')
st.write("""A real time scoring API to predict clients' repayment abilities""")

# load data
data_load_state = st.text('Loading data...')
data = load_data(data_path)
data_load_state.text("Loading data...completed")

# Let the user enter the client's id
st.subheader("Client's ID")
client_id = st.number_input('Insert a client ID',min_value = 100001,
                            max_value=500000, on_change=reinit_results)
client_data = data.loc[[client_id]]
json_data = client_data.to_json()

if 'get_results' not in st.session_state:
    st.session_state['get_results'] = False

# To automatically display results when changing a widget value
if st.button('Get predictions'):
    st.session_state['get_results'] = True

# Display results
if st.session_state['get_results']:
    st.header('Results')
    pred_state = st.text("Waiting for predictions...")
    score, probability = get_predictions(json_data)
    if score is None:
        pred_state.text("error: the prediction API request did not work") 
    else:
        pred_state.text("predictions successfully loaded")
        st.write('**Client:**', client_id)
        st.write('**Score:**', score)
        st.write('**probability:**', probability)  
    if score == 1:
        message = '<span style="font-size:18px; color:red">\
                    The client will probably be in **default**.</span>'
    else:
        message = '<span style="font-size:18px; color:green">\
                   The Client\'s default is **unlikely**</span>'
    st.markdown(message, unsafe_allow_html=True)

    # laying out the middle section of the app
    col_1, col_2 = st.columns((1, 1))
    with col_1:
        st.subheader('Prediction Interpretation Plot')
        # Get shap values
        means = np.mean(data.values, axis=0)
        shap_values, exp_value = get_shap_values(client_data, means)
        if shap_values is None:
            st.text("error: the explainer API request did not work") 
        else:
            f_plot = shap.force_plot(exp_value, shap_values, client_data)
            st_shap(f_plot)
        st.subheader('Summary Plot')
        fig = plt.figure()
        shap.summary_plot(shap_values, client_data, max_display=10, show=False)
        st.pyplot(fig)
    with col_2:
        # Get the top features
        st.subheader('Distribution plot')
        top_features = top_features(shap_values, feature_names)
        # Select a feature
        feature = st.selectbox('Select a feature to plot', top_features,
                               key='dis_feature')
        client_val = round(float(client_data[feature]), 2)
        dis_data = data[feature]
        # Plot the feature distribution
        st.write('client value', client_val)
        dis_plot = plt.figure()
        sns.histplot(dis_data, bins=20, kde=True)
        plt.axvline(client_val, c='r', ls=':', label='client value')
        plt.legend()
        st.pyplot(dis_plot)