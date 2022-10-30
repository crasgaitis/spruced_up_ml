import streamlit as st
import pandas as pd
import numpy as np
import math
from PIL import Image
import sklearn
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import gzip
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

with gzip.open('ml_model.pkl', 'rb') as f:
    p = pickle.Unpickler(f)
    clf = p.load()    

with open("pipeline.pkl", 'rb') as file2:
    full_pipeline = pickle.load(file2)

@st.cache
def load_data(file):
    data = pd.read_csv(file)
    return data

DATA = 'Trees.csv'
data = load_data(DATA)
data = data[['DIAM', 'SITETYPE', 'GENUS', 'PRIMARYDISTRICTCD']]

st.title("Find your tree's new home")

DIAM = st.selectbox("Diameter (inches)", ('2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17'))

SITETYPE = st.selectbox("Site type", ("STRP", "PIT"))

GENUS = st.selectbox("Genus", ("Acer", "Prunus", "Malus", "Quercus", "Pyrus"))

submit = st.button("Calculate")

user_input = pd.DataFrame(np.array([[DIAM, SITETYPE, GENUS]]), columns =['DIAM', 'SITETYPE', 'GENUS'])

input_ = {
    "DIAM": int(DIAM),
    "SITETYPE": [SITETYPE],
    "GENUS": [GENUS]
}
user_input = pd.DataFrame(input_) 

cat = ['DIAM', 'SITETYPE', 'GENUS']

if submit:
    user_input_prep = full_pipeline.transform(user_input)
    
    pred = clf.predict(user_input_prep)
    
    if pred == [6]:
         location = "District 7"
    elif pred == [5]:
         location = "District 6"
    elif pred == [4]:
         location = "District 5"
    elif pred == [3]:
         location = "District 4"
    elif pred == [2]:
         location = "District 3"
    elif pred == [1]:
         location = "District 2"
    else:
        location = "District 1"
        
    st.title(location)