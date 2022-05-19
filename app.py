#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
import pandas as pd
import pickle
import json
import shap

# read the model pickle file
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
feature_names = model['features_selected']
clf = model['classifier']

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
    return clf.predict_proba(X)[:, 1] > 0.2

# read pickle files
# The SHAP explainer is based on the predictor above
with open('explainer.pkl', 'rb') as exp_file:
    explainer = pickle.load(exp_file)

app = Flask(__name__)

@app.route('/')
def hello():
    # A welcome message to test the api
    return "<h1>Welcome to my prediction api!</h1>"

@app.route('/predictions/', methods=['POST'])
def get_predictions():
    content = request.get_json()
    data = pd.DataFrame(json.loads(content))
    # get the right features order
    data = data[feature_names]
    # get predictions
    score = int(score_predictor(data.values))
    probability = round(float(clf.predict_proba(data.values)[:, 1]), 2)
    return jsonify(score, probability)

@app.route('/shap/', methods=['POST'])
def get_shap_values():
    content = request.get_json()
    data = pd.DataFrame(json.loads(content))
    shap_values = explainer.shap_values(data).tolist()[0]
    return jsonify([shap_values, explainer.expected_value])

if __name__ == '__main__':
    app.run(port=5000)