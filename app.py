# app.py - Main Navigation Portal
import streamlit as st

st.sidebar.title("Navigation")
app_choice = st.sidebar.radio("Select Activation App:", ["1. ReLU", "2. Sigmoid", "3. Tanh"])

if app_choice == "1. ReLU":
    import app_relu
elif app_choice == "2. Sigmoid":
    import app_sigmoid
else:
    import app_tanh
