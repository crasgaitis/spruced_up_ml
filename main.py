import streamlit as st
import pandas as pd
import numpy as np
import math
from PIL import Image
import sklearn
## from pipeline import full_pipeline
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
    
@st.cache
def load_data(file):
    data = pd.read_csv(file)
    return data

DATA = 'Trees.csv'
data = load_data(DATA)
data = data[['DIAM', 'SITETYPE', 'GENUS', 'PRIMARYDISTRICTCD']]

st.title("Find your tree's new home")

max_DIAM = int(data["DIAM"].max())
min_DIAM = 0
DIAM = st.slider("Tree Diameter (inches)", min_DIAM, max_DIAM, int((min_DIAM+max_DIAM)/2))

SITETYPE = st.selectbox("Site type", ("STRP", "PIT"))

GENUS = st.selectbox("Genus", ("Acer", "Prunus", "Malus", "Crataegus", "Quercus", "Cornus", "Pyrus"))

submit = st.button("Calculate")

user_input = pd.DataFrame(np.array([[DIAM, SITETYPE, GENUS]]), columns =['DIAM', 'SITETYPE', 'GENUS'])
st.write(user_input)

num = ['DIAM']
cat = ['SITETYPE', 'GENUS']

num_pipeline = Pipeline([
  ("std_scaler", StandardScaler())
])

col_pipeline = ColumnTransformer([  
  ("num", num_pipeline, num), 
  ("cat", OneHotEncoder(), cat),  
])

full_pipeline = Pipeline([   
  ("col_pipeline", col_pipeline)
])


if submit:
    user_input_prep = full_pipeline.transform(user_input)
    st.write(user_input_prep)