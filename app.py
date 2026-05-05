import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# Load Dataset
# -------------------------
try:
    df = pd.read_csv("student_mental_health.csv", on_bad_lines='skip', engine='python')
    
    # CLEANUP: Remove any leading/trailing spaces from column names
    df.columns = df.columns.str.strip()
    
    # Drop rows with missing values
    df = df.dropna().copy()
except Exception as e:
    st.error(f"❌ Error loading CSV: {e}")
    st.stop()

st.title("🎓 Student Mental Health Predictor")

# -------------------------
# Select Target Column
# -------------------------
# We do this BEFORE encoding so we know what to exclude from X
target = st.selectbox("Select the column you want to predict", df.columns)

# -------------------------
# Encode Data
# -------------------------
label_encoders = {}
df_encoded = df.copy()

for col in df_encoded.columns:
    # If the column is text, convert it to numbers
    if df_encoded[col].dtype == 'object':
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

# -------------------------
# Split Data (The "X" fix)
# -------------------------
# Ensure X ONLY has numeric features and definitely does not have the target
X = df_encoded.drop(columns=[target])
y = df_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Train Model
# -------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

# -------------------------
# User Input (Dynamic UI)
# -------------------------
st.subheader("Enter Values for Prediction")
user_input = []

for col in X.columns:
    # If it was originally a text column, show a dropdown
    if col in label_encoders:
        options = list(label_encoders[col].classes_)
        selected = st.selectbox(f"Select {col}", options)
        user_input.append(label_encoders[col].transform([selected])[0])
    else:
        # Otherwise, show a number input
        avg_val = float(df[col].mean())
        user_input.append(st.number_input(f"Enter {col}", value=avg_val))

# -------------------------
# Prediction
# -------------------------
if st.button("Predict"):
    prediction = model.predict([user_input])[0]
    
    # If the target was a word (like 'Medium'), convert it back
    if target in label_encoders:
        final_result = label_encoders[target].inverse_transform([int(prediction)])[0]
    else:
        final_result = prediction
        
    st.success(f"The predicted **{target}** is: **{final_result}**")
