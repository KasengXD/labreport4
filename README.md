# BSD3513 Lab Report 4: Neural Network Activation Functions

## Project Description

This project develops three separate web applications using **Python and Streamlit** to demonstrate and compare three neural network activation functions:

1. ReLU (Rectified Linear Unit)
2. Sigmoid
3. Tanh (Hyperbolic Tangent)

Each application allows users to train a neural network model and observe how different activation functions affect the learning process and classification results.

## Features

- Select between ReLU, Sigmoid, and Tanh activation functions.
- Choose different datasets, including Moons and Circles.
- Adjust model parameters such as the number of samples, noise, hidden layers, learning rate, and random seed.
- Train the neural network using the **Train Model** button.
- View the training loss curve.
- View the model's decision boundary.
- Display training and testing accuracy.

## Project Structure

```text
BSD3513_Lab5/
│
├── app.py
├── app_relu.py
├── app_sigmoid.py
├── app_tanh.py
├── requirements.txt
└── README.md
```

## Technologies Used

- Python
- Streamlit
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn

## Installation

**Step 1: Clone the repository**

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

**Step 2: Navigate to the project folder**

```bash
cd BSD3513_Lab5
```

**Step 3: Install the required libraries**

```bash
pip install -r requirements.txt
```

Alternatively, install the libraries directly:

```bash
pip install streamlit torch numpy matplotlib scikit-learn
```

## Running the Application

To run the main application with the activation function selection menu:

```bash
streamlit run app.py
```

Users can select ReLU, Sigmoid, or Tanh from the sidebar.

To run an individual activation function application:

**ReLU**
```bash
streamlit run app_relu.py
```

**Sigmoid**
```bash
streamlit run app_sigmoid.py
```

**Tanh**
```bash
streamlit run app_tanh.py
```

## Activation Functions

| Activation Function | Description |
|---|---|
| ReLU | Returns zero for negative inputs and the input value for positive inputs. |
| Sigmoid | Converts input values into a range between 0 and 1. |
| Tanh | Converts input values into a range between -1 and 1. |

## Usage

1. Start the application using Streamlit.
2. Select an activation function from the sidebar.
3. Choose a dataset and adjust the model parameters.
4. Click the **Train Model** button.
5. Wait for the model training process to finish.
6. Examine the training loss, accuracy results, and decision boundary.

## Student Information

- **Course:** BSD3513 Introduction to Artificial Intelligence
- **Assignment:** Chapter 4 – Neural Networks, Lab Report 5
- **Student Name:** [WONG KA SENG]
- **Student ID:** [SD24037]

## Links

- **Deployed Application:** [https://labreport4-sd24037.streamlit.app/]
