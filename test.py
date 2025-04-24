import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load dataset from public URL
df = pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/Iris.csv")

# Rename and prepare
df.rename(columns={
    "sepal_length": "sepal_length", 
    "sepal_width": "sepal_width",
    "petal_length": "petal_length", 
    "petal_width": "petal_width",
    "species": "target"
}, inplace=True)

# Encode target
target_mapping = {"setosa": 0, "versicolor": 1, "virginica": 2}
df["target"] = df["target"].map(target_mapping)

# Define features and target
X = df.drop(columns=["target"])
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))

# Prediction function
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    target_names = ["Setosa", "Versicolor", "Virginica"]
    return target_names[prediction[0]]

# --- Streamlit UI ---

st.title("🌸 Iris Species Predictor")
st.caption("Powered by Logistic Regression and the classic Iris dataset")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)

# Predict button
if st.button("Classify"):
    species = predict_species(sepal_length, sepal_width, petal_length, petal_width)
    st.success(f"🌼 Predicted Species: **{species}**")

# Show model performance
st.markdown(f"**Model Accuracy:** `{accuracy * 100:.2f}%`")
