import streamlit as st
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# =========================
# LOAD DATA
# =========================
data = pd.read_csv("520 DATASET.csv")

X = data[['KI','KII','KIII']]
Y = data['Crack_Length_mm']

# =========================
# TRAIN MODEL (Decision Tree)
# =========================
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1
)

model = DecisionTreeRegressor()
model.fit(X_train, Y_train)

# Model performance
pred_test = model.predict(X_test)
r2 = r2_score(Y_test, pred_test)

# =========================
# CLASS MAPPING
# =========================
classes = np.array([5, 10, 15, 20])

def map_to_class(y):
    return classes[np.argmin(abs(classes - y))]

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Crack Length Predictor", layout="centered")

st.title("🔧 Crack Length Prediction using SIF (Decision Tree)")
st.write("Predict crack length from KI, KII, KIII values")

# INPUTS
KI = st.number_input("Enter KI", value=5.0)
KII = st.number_input("Enter KII", value=4.0)
KIII = st.number_input("Enter KIII", value=3.5)

# BUTTON
if st.button("Predict Crack Length"):

    input_data = np.array([[KI, KII, KIII]])
    
    pred = model.predict(input_data)[0]
    class_pred = map_to_class(pred)

    st.subheader("📊 Prediction Result")
    st.write(f"Predicted Crack Length: **{pred:.2f} mm**")
    st.write(f"Nearest Class: **{class_pred} mm**")

# =========================
# MODEL PERFORMANCE
# =========================
st.sidebar.title("Model Info")
st.sidebar.write(f"R² Score: {r2:.4f}")
st.sidebar.write("Model: Decision Tree Regressor")
