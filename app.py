
import streamlit as st
import importlib
import sys

# Page Configuration
st.set_page_config(
    page_title="BSD3513 Neural Network Lab",
    page_icon="🧠",
    layout="wide"
)

# Main Title
st.title("Neural Network Activation Function Lab")

st.write(
    """
    Welcome to the Neural Network Activation Function Visualizer.
    Select an activation function below to explore its graph,
    train a neural network and view its performance.
    """
)

# Sidebar Navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("### Select an Application")

app_options = {
    "ReLU — Rectified Linear Unit": "app_relu",
    "Sigmoid — Logistic Activation": "app_sigmoid",
    "Tanh — Hyperbolic Tangent": "app_tanh"
}

selected_app = st.sidebar.selectbox(
    "Choose Activation Function",
    options=list(app_options.keys()),
    index=0
)

# Application descriptions
descriptions = {
    "ReLU — Rectified Linear Unit":
        "ReLU outputs zero for negative inputs and keeps positive inputs unchanged.",

    "Sigmoid — Logistic Activation":
        "Sigmoid converts input values into a range between 0 and 1.",

    "Tanh — Hyperbolic Tangent":
        "Tanh converts input values into a range between -1 and 1."
}

st.sidebar.divider()

st.sidebar.markdown("### About Selected Function")
st.sidebar.info(descriptions[selected_app])

# Load Selected Application
module_name = app_options[selected_app]

if module_name in sys.modules:
    importlib.reload(sys.modules[module_name])
else:
    importlib.import_module(module_name)
