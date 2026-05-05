import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# 1. Load and Clean
try:
    df = pd.read_csv("student_mental_health.csv", on_bad_lines='skip', engine='python')
    # Remove hidden spaces in column names (e.g., " GPA " -> "GPA")
    df.columns = df.columns.str.strip()
    df = df.dropna().copy()
except Exception as e:
    st.error(f"❌ Error loading CSV: {e}")
    st.stop()

st.title("🎓 Student Mental Health Predictor")

# 2. Select Target
# This allows you to pick ANY column (GPA, Stress, etc.) as the goal
target = st.selectbox("What do you want to predict?", df.columns, index=len(df.columns)-1)

# 3. Handle Text, Ranges, and Categories
# We create a version of the data where EVERYTHING is a number
df_encoded = df.copy()
label_encoders = {}

for col in df_encoded.columns:
    # If the column has words or ranges (like "3.0 - 3.5"), we encode it
    if df_encoded[col].dtype == 'object':
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le
    else:
        # Ensure numeric columns are actually floats
        df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce').fillna(0)

# 4. Prepare the AI Model
X = df_encoded.drop(columns=[target])
y = df_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Create the Input Boxes (Dynamic UI)
st.subheader("Enter Student Details")
user_input = []

for col in X.columns:
    if col in label_encoders:
        # If it's a category (like GPA ranges or Stress Levels), show a dropdown
        options = list(label_encoders[col].classes_)
        choice = st.selectbox(f"{col}", options)
        user_input.append(label_encoders[col].transform([choice])[0])
    else:
        # If it's a simple number (like Age)
        avg = float(df[col].mean()) if col in df.columns else 0.0
        val = st.number_input(f"{col}", value=avg)
        user_input.append(val)

# 6. Run the Prediction
if st.button("Predict Now"):
    # Reshape input for the model
    features = np.array([user_input])
    prediction_num = model.predict(features)[0]
    
    # Convert the numeric answer back to words if needed
    if target in label_encoders:
        final_result = label_encoders[target].inverse_transform([int(prediction_num)])[0]
    else:
        final_result = round(prediction_num, 2)
        
    st.success(f"The predicted **{target}** is: **{final_result}**")
