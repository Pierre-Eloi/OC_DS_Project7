#! /usr/bin/env python3
# coding: utf-8

#import dill as pickle
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap

# Setting page config to wide mode
st.set_page_config(layout="wide")

st.title('Python Real Time Scoring API with Home Credit data')
# description and instructions
st.write("""A real time scoring API to predict clients' repayment abilities.\n
Enter the **client's ID** and click on `Get predictions`""")

data_path = 'data/application_test.csv'

@st.cache
def load_data():
    data = (pd.read_csv(data_path)
              .set_index('SK_ID_CURR'))
    #means = np.mean(transformer.transform(data), axis=0).reshape(1, -1)
    #return data, means
    return data

# Let the user input the client's id
#st.header('User Input features')
client_id = st.number_input('Insert the client ID',min_value = 100001,
                            max_value=500000)

# read the model pickle file
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
transformer = model['transformer']
features_selected = model['features_selected']
classifier = model['classifier']

# Create a predictor returning a 90% Recall
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
def explain_model(X, feature_names):
    # Calculate Shap values
    shap_values = explainer.shap_values(X)
    p = shap.force_plot(explainer.expected_value, shap_values,
                        X, feature_names=feature_names)
    return p, shap_values

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

submit = st.button('Get predictions')
if submit:
    # Let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
    #data, means = load_data()
    data = load_data()
    client_data = data.loc[[client_id]]
    X = transformer.transform(client_data)
    # Notify the reader that the data was successfully loaded.
    data_load_state.text("Loading data...completed")
    # Display results
    st.subheader('Results')
    prediction = int(score_predictor(X))
    probability = classifier.predict_proba(X)[:, 1]
    st.write('Client', client_id)
    st.write('Score', prediction)
    st.write('Probability', round(float(probability), 2))
    if prediction == 1:
        st.write('The client will probably be **in default**')
    else:
        st.write('Client default is **unlikely**')
    # SHAP force plot
    st.subheader('Model Prediction Interpretation Plot')
    p, shap_values = explain_model(X, features_selected)
    st_shap(p)
    # laying out the middle section of the app
    col_1, col_2 = st.columns((1, 1))
    with col_1:
        # Summary plot
        st.subheader('Summary Plot')
        fig = plt.figure()
        shap.summary_plot(shap_values, X, feature_names=features_selected,
        max_display=10, show=False)
        st.pyplot(fig)
    with col_2:
        # get the top features
        st.subheader('Distribution plot')
        top_features = top_features(shap_values, features_selected)
        st.selectbox('Select a feature to plot', top_features)
