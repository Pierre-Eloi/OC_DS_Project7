#! /usr/bin/env python3
# coding: utf-8

import streamlit as st
import streamlit.components.v1 as components
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
data_path = '../datasets/application_test.pqt'

# read the model pickle file
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
transformer = model['transformer']
features_selected = model['features_selected']
classifier = model['classifier']

@st.experimental_memo # replace @st.cache, do refer to the doc for clarifications
def load_data(feature_names):
    data = (pd.read_parquet(data_path)
              .set_index('SK_ID_CURR'))
    X = transformer.transform(data)
    return pd.DataFrame(X, columns=feature_names, index=data.index)

data_load_state = st.text('Loading data...')
data = load_data(features_selected)
data_load_state.text("Loading data...completed")

if 'get_results' not in st.session_state:
    st.session_state['get_results'] = False

def reinit_results():
    st.session_state['get_results'] = False

# Let the user enter the client's id
st.subheader("Client's ID")
client_id = st.number_input('Insert a client ID',min_value = 100001,
                            max_value=500000, on_change=reinit_results)
client_data = data.loc[[client_id]]


# Create a predictor returning a 90% Recall
@st.experimental_memo
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
# Import a SHAP explainer based on the predictor above
with open('explainer.pkl', 'rb') as file:
    explainer = pickle.load(file)

# Plot a force plot in streamlit API
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# Explain model prediction results with a force plot
@st.experimental_memo
def explain_model(X):
    # Calculate Shap values
    shap_values = explainer.shap_values(X)
    f_plot = shap.force_plot(explainer.expected_value, shap_values, X)
    return f_plot, shap_values

# Get the list of top features
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
              .iloc[:10, 0]
              .to_list())


# To automatically display results when changing a widget value
if st.button('Get predictions'):
    st.session_state['get_results'] = True

# Display results
if st.session_state['get_results']:
    st.header('Results')
    prediction = int(score_predictor(client_data))
    probability = classifier.predict_proba(client_data)[:, 1]
    st.write('Client', client_id)
    st.write('Score', prediction)
    st.write('Probability', round(float(probability), 2))
    if prediction == 1:
        st.write('The client will probably be **in default**')
    else:
        st.write('Client default is **unlikely**')
    # laying out the middle section of the app
    col_1, col_2 = st.columns((1, 1))
    with col_1:
        # SHAP force plot
        st.subheader('Model Prediction Interpretation Plot')
        p, shap_values = explain_model(client_data)
        st_shap(p)
        # Summary plot
        st.subheader('Summary Plot')
        fig = plt.figure()
        shap.summary_plot(shap_values, client_data, max_display=10, show=False)
        st.pyplot(fig)
        with col_2:
            # Get the top features
            st.subheader('Distribution plot')
            top_features = top_features(shap_values, features_selected)
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
