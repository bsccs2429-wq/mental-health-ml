import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Mental Health Predictor", layout="wide")

# 1. Load and Clean (Cached for speed)
@st.cache_data
def load_data():
    data = pd.read_csv("student_mental_health.csv", on_bad_lines='skip', engine='python')
    data.columns = data.columns.str.strip()
    return data.dropna()

df = load_data()

st.title("🎓 Student Mental Health Predictor")

# 2. Select Target
target = st.sidebar.selectbox("🎯 Target Variable", df.columns, index=len(df.columns)-1)

# 3. Robust Encoding
df_encoded = df.copy()
label_encoders = {}

for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

# 4. Prepare & Train Model (Done once at startup)
X = df_encoded.drop(columns=[target])
y = df_encoded[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 5. User Input Form (The UI Fix)
st.subheader("📝 Enter Student Information")
with st.form("prediction_form"):
    # Create two columns for a cleaner look
    cols = st.columns(2)
    user_input_raw = []
    
    for i, col_name in enumerate(X.columns):
        with cols[i % 2]: # Alternate between columns
            if col_name in label_encoders:
                options = list(label_encoders[col_name].classes_)
                choice = st.selectbox(f"{col_name}", options)
                user_input_raw.append(label_encoders[col_name].transform([choice])[0])
            else:
                avg = float(df[col_name].mean())
                val = st.number_input(f"{col_name}", value=avg)
                user_input_raw.append(val)
    
    # The Submit Button
    submit_button = st.form_submit_button(label="Predict Output")

# 6. Logic when Button is Clicked
if submit_button:
    # This block ONLY runs when the button is pressed
    try:
        prediction_num = model.predict([user_input_raw])[0]
        
        # Decode result if it was categorical
        if target in label_encoders:
            final_result = label_encoders[target].inverse_transform([int(prediction_num)])[0]
        else:
            final_result = f"{prediction_num:.2f}"
            
        st.divider()
        st.balloons() # Visual celebration
        st.success(f"### Result: The predicted **{target}** is **{final_result}**")
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")
