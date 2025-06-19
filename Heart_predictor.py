import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def user_input_features():
    age = st.slider('Age', 20, 80, 50)
    sex = st.selectbox('Sex (1=Male, 0=Female)', [0, 1])
    cp = st.selectbox('Chest Pain Type (0-3)', [0, 1, 2, 3])
    trestbps = st.slider('Resting Blood Pressure', 80, 200, 120)
    chol = st.slider('Cholesterol', 100, 600, 200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)', [0, 1])
    restecg = st.selectbox('Resting ECG Results (0-2)', [0, 1, 2])
    thalach = st.slider('Max Heart Rate Achieved', 70, 210, 150)
    exang = st.selectbox('Exercise Induced Angina (1=Yes, 0=No)', [0, 1])
    oldpeak = st.slider('ST Depression Induced by Exercise', 0.0, 6.0, 1.0)
    slope = st.selectbox('Slope of Peak Exercise ST Segment (0-2)', [0, 1, 2])
    ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy (0-4)', [0, 1, 2, 3, 4])
    thal = st.selectbox('Thalassemia (1=Normal, 2=Fixed Defect, 3=Reversable Defect)', [1, 2, 3])

    features = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    return pd.DataFrame(features, index=[0])

st.title("Heart Disease Prediction App ❤️")
uploaded_file = st.file_uploader("Upload your Heart Disease CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    X = data_imputed.drop('target', axis=1)
    y = data_imputed['target']

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(probability=True),
        "Logistic Regression": LogisticRegression(),
        "Multilayer Perceptron": MLPClassifier(max_iter=1000)
    }

    accuracies = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        accuracies[name] = acc

    best_model_name = max(accuracies, key=accuracies.get)
    best_model = models[best_model_name]

    st.subheader(f"Using the Best Model: {best_model_name}")

    input_df = user_input_features()
    input_scaled = scaler.transform(input_df)
    prediction = best_model.predict(input_scaled)
    prediction_proba = best_model.predict_proba(input_scaled) if hasattr(best_model, "predict_proba") else None

    if st.button("Predict"):
        st.markdown('---')
        st.write('#### Prediction')
        result = ['No Heart Disease', 'Heart Disease']
        color = 'green' if prediction[0] == 0 else 'red'
        st.markdown(
            f"<h2 style='text-align: center; color:{color};'>{result[prediction[0]]}</h2>",
            unsafe_allow_html=True
        )
        st.markdown('---')
        if prediction_proba is not None:
            st.write('#### Prediction Probability')
            st.subheader(f"Probability of No Heart Disease: {prediction_proba[0][0]:.2f}")
            st.subheader(f"Probability of Heart Disease: {prediction_proba[0][1]:.2f}")
