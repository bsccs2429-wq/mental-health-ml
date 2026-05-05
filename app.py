import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# 1. Load and Clean Column Names
try:
    df = pd.read_csv("student_mental_health.csv", on_bad_lines='skip', engine='python')
    # This removes hidden spaces like " study_hours " which cause selection errors
    df.columns = df.columns.str.strip() 
    df = df.dropna().copy()
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

st.title("🎓 Student Mental Health Predictor")

# 2. Setup Target
# Default to the last column if user hasn't picked one
target = st.selectbox("Select what you want to predict", df.columns, index=len(df.columns)-1)

# 3. THE FIX: Force everything to be numeric
df_encoded = df.copy()
label_encoders = {}

for col in df_encoded.columns:
    # If the column is text (object), we MUST encode it to numbers
    if df_encoded[col].dtype == 'object':
        le = LabelEncoder()
        # astype(str) ensures that "Medium" and 1.0 in the same column don't crash it
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

# 4. Prepare Features (X) and Labels (y)
# We use df_encoded because it is now 100% numbers
X = df_encoded.drop(columns=[target])
y = df_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model Training
model = RandomForestClassifier()
# This is where the crash was happening. Now X_train is guaranteed to be floats/ints.
model.fit(X_train, y_train)

# 6. Dynamic User Inputs
st.subheader("Enter Details for Prediction")
user_input = []

for col in X.columns:
    if col in label_encoders:
        # Show the original words (High, Low) but send numbers to the model
        options = list(label_encoders[col].classes_)
        choice = st.selectbox(f"Select {col}", options)
        user_input.append(label_encoders[col].transform([choice])[0])
    else:
        # For numeric columns like Age or Hours
        val = st.number_input(f"Enter {col}", value=float(df[col].mean()))
        user_input.append(val)

# 7. Prediction Logic
if st.button("Predict"):
    prediction_raw = model.predict([user_input])[0]
    
    # If the answer is a category (like 'High'), convert the number back to the word
    if target in label_encoders:
        result = label_encoders[target].inverse_transform([int(prediction_raw)])[0]
    else:
        result = prediction_raw
        
    st.success(f"The predicted **{target}** is: **{result}**")
