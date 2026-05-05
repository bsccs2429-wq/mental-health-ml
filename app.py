import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Mental Health Predictor", layout="wide")

# 1. Load Data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("student_mental_health.csv", on_bad_lines='skip', engine='python')
        data.columns = data.columns.str.strip() # Clean names
        return data.dropna()
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return None

df = load_data()

if df is not None:
    st.title("🎓 Student Mental Health Predictor")

    # 2. Select Target
    target = st.sidebar.selectbox("🎯 Select what to predict", df.columns, index=len(df.columns)-1)

    # 3. Robust Encoding Logic
    df_encoded = df.copy()
    label_encoders = {}

    for col in df_encoded.columns:
        # If the column is text OR contains ranges, force it to numeric categories
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
        else:
            # Ensure numbers are actually numbers
            df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce').fillna(0)

    # 4. Prepare Features and Model
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Attempt to train - if this fails, we show the data types to debug
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        st.error("🚨 Model Training Error!")
        st.write("The model found non-numeric data. Here are the types it sees:")
        st.write(X_train.dtypes)
        st.stop()

    # 5. The Input Form
    st.subheader(f"Predicting: {target}")
    with st.form("main_form"):
        inputs = []
        cols = st.columns(2)
        
        for i, col_name in enumerate(X.columns):
            with cols[i % 2]:
                if col_name in label_encoders:
                    # Dropdown for category/text columns
                    options = list(label_encoders[col_name].classes_)
                    choice = st.selectbox(f"{col_name}", options)
                    inputs.append(label_encoders[col_name].transform([choice])[0])
                else:
                    # Number box for numeric columns
                    val = st.number_input(f"{col_name}", value=float(df[col_name].mean()))
                    inputs.append(val)
        
        submit = st.form_submit_button("Generate Prediction")

    # 6. Output
    if submit:
        prediction = model.predict([inputs])[0]
        
        # Convert back to text if it was a category
        if target in label_encoders:
            result = label_encoders[target].inverse_transform([int(prediction)])[0]
        else:
            result = round(float(prediction), 2)
            
        st.divider()
        st.balloons()
        st.success(f"### Predicted {target}: **{result}**")
        
        # Quick Insight
        st.info("The model used Random Forest logic to determine this result based on your dataset patterns.")
