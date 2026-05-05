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
    
    # CRITICAL FIX: Remove hidden spaces from column names
    df.columns = df.columns.str.strip()
    
    # Drop empty rows and work on a clean copy
    df = df.dropna().copy()
except Exception as e:
    st.error(f"❌ CSV Error: {e}")
    st.stop()

st.title("🎓 Student Mental Health Predictor")

# -------------------------
# Dataset Preview
# -------------------------
st.subheader("Dataset Preview")
st.write(df.head())

# -------------------------
# Select Target Column
# -------------------------
# Now "study_hours" (without spaces) will be selectable
target = st.selectbox("Select the column you want to predict", df.columns)

# -------------------------
# Encode Categorical Data
# -------------------------
label_encoders = {}
df_encoded = df.copy()

for col in df_encoded.columns:
    # Check if column contains text (object)
    if df_encoded[col].dtype == 'object':
        le = LabelEncoder()
        # Convert to string first to be safe, then encode
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

# -------------------------
# Split Data
# -------------------------
# We use the ENCODED version for the machine learning model
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
    # If the column was text, show a dropdown menu
    if col in label_encoders:
        options = list(label_encoders[col].classes_)
        selected_option = st.selectbox(f"Select {col}", options)
        # Convert selection back to the number the model expects
        encoded_val = label_encoders[col].transform([selected_option])[0]
        user_input.append(encoded_val)
    else:
        # For numeric columns, use a number box
        default_val = float(df[col].mean())
        val = st.number_input(f"Enter {col}", value=default_val)
        user_input.append(val)

# -------------------------
# Prediction Logic
# -------------------------
if st.button("Predict"):
    features = np.array([user_input])
    prediction = model.predict(features)[0]

    # Convert numeric prediction back to its original name (e.g., "High")
    if target in label_encoders:
        final_result = label_encoders[target].inverse_transform([int(prediction)])[0]
    else:
        final_result = prediction

    st.success(f"The predicted **{target}** is: **{final_result}**")
