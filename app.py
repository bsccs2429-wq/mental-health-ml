import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# 1. Load and Clean
try:
    df = pd.read_csv("student_mental_health.csv", on_bad_lines='skip', engine='python')
    # Clean hidden spaces in column names
    df.columns = df.columns.str.strip()
    df = df.dropna().copy()
except Exception as e:
    st.error(f"❌ Error loading CSV: {e}")
    st.stop()

st.title("🎓 Student Mental Health Predictor")

# 2. Select Target
target = st.selectbox("Select target variable", df.columns, index=len(df.columns)-1)

# 3. Robust Encoding
# This part is key: it treats everything as a category if it's not a pure number
df_encoded = df.copy()
label_encoders = {}

for col in df_encoded.columns:
    # Force to string first to handle ranges like "3.50 - 4.00" or course names
    if df_encoded[col].dtype == 'object':
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

# 4. Prepare Features
X = df_encoded.drop(columns=[target])
y = df_encoded[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 6. User Inputs (Dynamic Logic)
st.subheader("Predictor Inputs")
user_input = []

for col in X.columns:
    if col in label_encoders:
        # Show dropdowns for text/range columns (like CGPA)
        options = list(label_encoders[col].classes_)
        choice = st.selectbox(f"{col}", options)
        user_input.append(label_encoders[col].transform([choice])[0])
    else:
        # Show number inputs for numeric columns (like Age)
        default_val = float(df[col].mean())
        val = st.number_input(f"{col}", value=default_val)
        user_input.append(val)

# 7. Prediction
if st.button("Predict"):
    prediction_numeric = model.predict([user_input])[0]
    
    # Decode result if it was a text category
    if target in label_encoders:
        final_result = label_encoders[target].inverse_transform([int(prediction_numeric)])[0]
    else:
        final_result = prediction_numeric
        
    st.success(f"The predicted **{target}** is: **{final_result}**")
