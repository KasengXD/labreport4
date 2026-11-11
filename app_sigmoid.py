# app_sigmoid.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Page Configuration
st.set_page_config(page_title="Sigmoid Activation NN Visualizer", layout="wide")

st.title("2. Sigmoid Activation Function")
st.write(
    """
    **Sigmoid** maps inputs to $(0, 1)$ range: $\\sigma(z) = \\frac{1}{1 + e^{-z}}$.
    This app visualizes PyTorch MLP performance using **Sigmoid** activation.
    """
)

# Sidebar hyperparameter controls
st.sidebar.header("Hyperparameters")
dataset_type = st.sidebar.selectbox("Dataset type", ["Moons", "Circles"])
n_samples = st.sidebar.slider("Number of samples", 200, 1500, 500, step=100)
noise = st.sidebar.slider("Noise level", 0.0, 0.5, 0.2, step=0.05)
hidden_size = st.sidebar.slider("Hidden layer size", 4, 128, 16, step=4)
n_hidden_layers = st.sidebar.slider("Number of hidden layers", 1, 3, 2)
learning_rate = st.sidebar.select_slider("Learning rate", options=[0.001, 0.003, 0.01, 0.03, 0.1], value=0.01)
epochs = st.sidebar.slider("Epochs", 10, 500, 150, step=10)
seed = st.sidebar.number_input("Random seed", 0, 9999, 42)

# Data preparation
torch.manual_seed(seed)
np.random.seed(seed)

if dataset_type == "Moons":
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
else:
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=seed)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=seed, stratify=y
)

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = torch.utils.data.DataLoader(SimpleDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(SimpleDataset(X_test, y_test), batch_size=64, shuffle=False)

# Sigmoid PyTorch Network
class SigmoidNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Sigmoid()]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Sigmoid())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = SigmoidNet(2, hidden_size, 2, n_hidden_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Section 1: Activation Curve
st.subheader("1. Sigmoid Curve & Derivative")
col_curve1, col_curve2 = st.columns(2)
z_vals = np.linspace(-6, 6, 400)
sig_vals = 1.0 / (1.0 + np.exp(-z_vals))
sig_grad = sig_vals * (1.0 - sig_vals)

with col_curve1:
    fig_act, ax_act = plt.subplots(figsize=(5, 3))
    ax_act.plot(z_vals, sig_vals, label="Sigmoid(z)", color="green", lw=2)
    ax_act.plot(z_vals, sig_grad, label="d/dz Sigmoid(z)", color="orange", linestyle="--")
    ax_act.set_title("Sigmoid Function and Derivative")
    ax_act.grid(True); ax_act.legend()
    st.pyplot(fig_act)

with col_curve2:
    st.markdown("""
    **Sigmoid Properties:**
    - Formula: $\\sigma(z) = \\frac{1}{1 + e^{-z}}$
    - Output Range: $(0, 1)$ — ideal for output probabilities.
    - Max gradient is $0.25$ at $z=0$.
    - Prone to **vanishing gradient problem** when $|z|$ is large.
    """)

# Section 2: Model Training
st.subheader("2. Model Training")
if st.button("Train Model with Sigmoid"):
    model.train()
    loss_history = []
    for epoch in range(epochs):
        running_loss = 0.0
        for b_x, b_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(b_x), b_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * b_x.size(0)
        loss_history.append(running_loss / len(train_loader.dataset))

    model.eval()
    def get_acc(loader):
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in loader:
                preds = model(bx).argmax(dim=1)
                correct += (preds == by).sum().item()
                total += by.size(0)
        return correct / total

    train_acc = get_acc(train_loader)
    test_acc = get_acc(test_loader)

    col1, col2 = st.columns(2)
    with col1:
        fig_loss, ax_loss = plt.subplots(figsize=(5, 3))
        ax_loss.plot(range(1, epochs + 1), loss_history, color="green")
        ax_loss.set_title("Training Loss Curve")
        ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss"); ax_loss.grid(True)
        st.pyplot(fig_loss)
    with col2:
        st.metric("Train Accuracy", f"{train_acc * 100:.2f}%")
        st.metric("Test Accuracy", f"{test_acc * 100:.2f}%")

    # Decision Boundary Plot
    st.subheader("3. Decision Boundary")
    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_tensor = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()

    with torch.no_grad():
        Z = model(grid_tensor).argmax(dim=1).numpy().reshape(xx.shape)

    fig_db, ax_db = plt.subplots(figsize=(6, 4))
    ax_db.contourf(xx, yy, Z, alpha=0.4, cmap="PiYG")
    ax_db.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, edgecolor="k", cmap="PiYG", s=25)
    ax_db.set_title("Sigmoid Decision Boundary")
    st.pyplot(fig_db)