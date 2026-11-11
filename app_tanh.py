
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

if __name__ == "__main__":
    st.set_page_config(page_title="Tanh Activation Function", page_icon="🧠", layout="wide")


def run_app():
    st.title("Tanh Activation Function")
    st.write("Tanh converts input values into a range between -1 and 1.")
    st.latex(r"f(x)=\tanh(x)")
    st.latex(r"f'(x)=1-f(x)^2")

    st.sidebar.header("Curve")
    x_input = st.sidebar.slider("Input value x", -10.0, 10.0, 1.0, 0.1, key="tanh_x")

    output = np.tanh(x_input)
    derivative = 1 - output ** 2

    st.sidebar.divider()
    st.sidebar.header("Neural Network Demo")

    dataset_name = st.sidebar.selectbox("Dataset", ["Moons", "Circles"], key="tanh_dataset")
    samples = st.sidebar.slider("Samples", 100, 1000, 300, 50, key="tanh_samples")
    noise = st.sidebar.slider("Noise", 0.0, 0.5, 0.2, 0.05, key="tanh_noise")

    with st.sidebar.expander("Advanced settings"):
        hidden_size = st.number_input("Hidden layer size", 4, 128, 16, 4, key="tanh_hidden")
        hidden_layers = st.number_input("Number of hidden layers", 1, 5, 2, key="tanh_layers")
        learning_rate = st.number_input("Learning rate", 0.0001, 0.1, 0.01, 0.001, format="%.4f", key="tanh_lr")
        epochs = st.slider("Max iterations (epochs)", 100, 1000, 800, 100, key="tanh_epochs")
        seed = st.number_input("Random seed", 1, 9999, 42, key="tanh_seed")

    st.header("1. Tanh Curve")
    col1, col2 = st.columns(2)

    x_values = np.linspace(-10, 10, 400)
    y_values = np.tanh(x_values)
    dy_values = 1 - y_values ** 2

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x_values, y_values, label="Tanh f(x)")
        ax.plot(x_values, dy_values, "--", label="Derivative f'(x)")
        ax.scatter(x_input, output, color="red", label="Your input", zorder=5)
        ax.set_title("Tanh Activation")
        ax.set_xlabel("Input x")
        ax.set_ylabel("Output")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Output for your input")
        c1, c2, c3 = st.columns(3)
        c1.metric("Input x", f"{x_input:.2f}")
        c2.metric("Tanh(x)", f"{output:.4f}")
        c3.metric("Derivative", f"{derivative:.4f}")
        st.subheader("Properties")
        st.write("- Range: (-1, 1)")
        st.write("- Output is centred around zero.")
        st.write("- Produces a smooth S-shaped curve.")
        st.write("- Gradient becomes small for extreme inputs.")

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    if dataset_name == "Moons":
        X, y = make_moons(n_samples=samples, noise=noise, random_state=int(seed))
    else:
        X, y = make_circles(n_samples=samples, noise=noise, factor=0.5, random_state=int(seed))

    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=int(seed), stratify=y
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    class TanhNet(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            input_size = 2
            for _ in range(int(hidden_layers)):
                layers.extend([nn.Linear(input_size, int(hidden_size)), nn.Tanh()])
                input_size = int(hidden_size)
            layers.extend([nn.Linear(input_size, 1), nn.Sigmoid()])
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    config = (dataset_name, samples, noise, hidden_size, hidden_layers, learning_rate, seed)
    state_key = "tanh_training_state"

    if state_key not in st.session_state or st.session_state[state_key]["config"] != config:
        torch.manual_seed(int(seed))
        model = TanhNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        st.session_state[state_key] = {
            "config": config, "model": model, "optimizer": optimizer,
            "loss_history": [], "trained": False
        }

    state = st.session_state[state_key]
    model = state["model"]
    optimizer = state["optimizer"]

    st.header("2. Neural Network Training")
    if st.button("Train Model", type="primary", key="tanh_train"):
        loss_function = nn.BCELoss()
        progress = st.progress(0)
        status = st.empty()

        for epoch in range(int(epochs)):
            model.train()
            total_loss = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = loss_function(predictions, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            state["loss_history"].append(total_loss / len(train_loader))
            progress.progress((epoch + 1) / int(epochs))
            if (epoch + 1) % 100 == 0:
                status.write(f"Training epoch {epoch + 1}/{epochs}")

        state["trained"] = True
        status.success("Model training completed!")

    if state["trained"]:
        st.header("3. Training Results")
        model.eval()

        with torch.no_grad():
            train_acc = ((model(X_train) >= 0.5) == y_train).float().mean().item() * 100
            test_acc = ((model(X_test) >= 0.5) == y_test).float().mean().item() * 100

        c1, c2 = st.columns(2)
        c1.metric("Training Accuracy", f"{train_acc:.2f}%")
        c2.metric("Testing Accuracy", f"{test_acc:.2f}%")

        st.subheader("Training Loss Curve")
        fig, ax = plt.subplots()
        ax.plot(state["loss_history"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Decision Boundary")
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))
        grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

        with torch.no_grad():
            Z = model(grid).numpy().reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3)
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="black")
        ax.set_title("Tanh Decision Boundary")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Adjust the hyperparameters and click Train Model to begin.")


if __name__ == "__main__":
    run_app()
