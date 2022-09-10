import streamlit as st
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
from scripts.modeling import Model
from scripts.feature_eng import FeatureEngineering

random_forest_args = {
    'n_estimators': 100,
    'max_depth': 16,
    'max_features': 'sqrt',
    'n_jobs': 4
}

def predict(data):
    loaded_model = pickle.load(open('../models/10-09-2022-10-14-42.pkl', 'rb'))
    feature_eng = FeatureEngineering(data)
    feature_eng.preprocess()
    features = feature_eng.df
    model = Model(features, random_forest_args)
    features = model.preprocess()
    y_pred = loaded_model.predict(features[2])
    fig, ax = plt.subplots()
    plt.plot(y_pred)
    st.pyplot(fig)
    

st.title('Pharmaceutical Sales Prediction')
uploadedfile = st.file_uploader("Upload csv")
infer_uploaded = st.button('Infer with the Uploaded data')

if (infer_uploaded):
    print(uploadedfile)
    with open(os.path.join("files",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
    data = pd.read_csv('files/'+uploadedfile.name)
    predict(data)
    
