import streamlit as st
import sklearn
import pandas as pd
import numpy as np
import math
from PIL import Image
import pickle
# import joblib
import random

DATA = 'Trees.csv'

@st.cache
def load_data(file):
    data = pd.read_csv(file)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data

