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



app = Flask(__name__)

@app.route('/')
def hello():
    # A welcome message to test the api
    return "<h1>Welcome to my prediction api!</h1>"

if __name__ == '__main__':
    app.run(port=5000)