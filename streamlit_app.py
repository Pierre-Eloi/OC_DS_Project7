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
