
import streamlit as st
import importlib

st.set_page_config(
    page_title="BSD3513 Neural Network Lab",
    page_icon="🧠",
    layout="wide"
)

st.title("Neural Network Activation Function Lab")

st.write(
    "Explore and compare three neural network activation functions "
    "through interactive curves and model training."
)

st.sidebar.title("Navigation")

app_options = {
    "ReLU — Rectified Linear Unit": "app_relu",
    "Sigmoid — Logistic Activation": "app_sigmoid",
    "Tanh — Hyperbolic Tangent": "app_tanh"
}

selected_app = st.sidebar.selectbox(
    "Choose Activation Function",
    options=list(app_options.keys())
)

descriptions = {
    "ReLU — Rectified Linear Unit":
        "ReLU returns zero for negative inputs and keeps positive inputs unchanged.",
    "Sigmoid — Logistic Activation":
        "Sigmoid converts input values into a range between 0 and 1.",
    "Tanh — Hyperbolic Tangent":
        "Tanh converts input values into a range between -1 and 1."
}

st.sidebar.divider()
st.sidebar.subheader("About Selected Function")
st.sidebar.info(descriptions[selected_app])

module = importlib.import_module(app_options[selected_app])
importlib.reload(module)
module.run_app()
