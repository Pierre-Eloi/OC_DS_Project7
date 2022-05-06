#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
import json
import shap

# read the model pickle file
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
transformer = model['transformer']
features_selected = model['features_selected']
classifier = model['classifier']

app = Flask(__name__)

@app.route('/api/predictions')
def get_prediction():
    json_data = request.get_json()
    #read the real time input to pandas df
    data = pd.DataFrame(json_data)
    #transform DataFrame
    data_tr = pd.DataFrame(transformer.transform(data),
                           columns=features_selected,
                           index=data.index)
    data_tr['probability'] = classifier.predict_proba(data_tr)[:, 1]
    return data_tr.to_json() 

if __name__ == '__main__':
    app.run( port=5000)