import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# Load Dataset (safe loading)
# -------------------------
try:
    df = pd.read_csv("student_mental_health.csv", on_bad_lines='skip', engine='python')
except:
    st.error("❌ CSV file not found or broken.")
    st.stop()

# -------------------------
# Show Data
# -------------------------
st.title("🎓 Student Mental Health Predictor")

st.subheader("Dataset Preview")
st.write(df.head())

st.subheader("Columns in Dataset")
st.write(list(df.columns))

# -------------------------
# Select Target Column
# -------------------------
target = st.selectbox("Select the column you want to predict", df.columns)

# -------------------------
# Encode Data
# -------------------------
df = df.dropna()

label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# -------------------------
# Split Data
# -------------------------
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Train Model
# -------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

# -------------------------
# User Input (dynamic)
# -------------------------
st.subheader("Enter Values")

user_input = []

for col in X.columns:
    val = st.number_input(f"{col}", value=0.0)
    user_input.append(val)

# -------------------------
# Prediction
# -------------------------
if st.button("Predict"):
    user_data = np.array([user_input])
    prediction = model.predict(user_data)[0]

    # Decode if needed
    if target in label_encoders:
        prediction = label_encoders[target].inverse_transform([prediction])[0]

    st.success(f"Prediction: {prediction}")