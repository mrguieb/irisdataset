import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# Load the dataset from CSV
df = pd.read_csv("iris.csv")  # Ensure the correct file path

# Rename columns to match the script
df.rename(columns={"SepalLengthCm": "sepal_length", "SepalWidthCm": "sepal_width",
                   "PetalLengthCm": "petal_length", "PetalWidthCm": "petal_width",
                   "Species": "target"}, inplace=True)

# Mapping target values to class names if they are stored as text
target_mapping = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
df["target"] = df["target"].map(target_mapping)

# Define feature columns and target
X = df.drop(columns=["target", "Id"])  # Drop "Id" column if present
y = df["target"]

# Train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Function to predict species
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    predicted_class = int(round(prediction[0]))  # Round to nearest class index
    predicted_class = np.clip(predicted_class, 0, 2)  # Ensure within valid range
    target_names = [" Setosa", " versicolor", " virginica"]
    return target_names[predicted_class]

# Streamlit UI
st.title("Iris Species Predictor using Linear Regression")

# User input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)

if st.button("Classify"):
    species = predict_species(sepal_length, sepal_width, petal_length, petal_width)
    st.success(f"Predicted Species: {species}")
