import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Startup Success Predictor", layout="wide")

st.title("🚀 Startup Success Predictor")
st.markdown(
    """
    This app predicts if a startup will succeed based on various features.
    Enter startup details below and get a prediction!
    """
)

@st.cache_data
def load_data():
    df = pd.read_csv('your_startup_dataset.csv')

    # Drop unneeded columns
    df = df.drop(columns=['Unnamed: 0', 'Unnamed: 6', 'object_id', 'state_code.1'], errors='ignore')

    # Fill missing numeric columns
    df.fillna({'funding_total_usd': 0, 'funding_rounds': 0, 'milestones': 0}, inplace=True)
    df.dropna(subset=['status'], inplace=True)

    # Convert target to binary: success=1 else 0
    df['status'] = df['status'].apply(lambda x: 1 if x in ['acquired', 'operating', 'ipo'] else 0)

    # Encode categorical features
    for col in ['city', 'category_code', 'state_code']:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')
            le = LabelEncoder()
            df[col + '_enc'] = le.fit_transform(df[col])
        else:
            # if col not present, create dummy column
            df[col + '_enc'] = 0

    return df

df = load_data()

# Features and target
feature_cols = ['funding_total_usd', 'funding_rounds', 'milestones',
                'has_VC', 'has_angel', 'has_roundA', 'has_roundB', 'has_roundC', 'has_roundD',
                'avg_participants', 'is_top500',
                'city_enc', 'category_code_enc', 'state_code_enc']

X = df[feature_cols]
y = df['status']

# Split data for training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.sidebar.header("Input Startup Details")

def user_input_features():
    funding_total_usd = st.sidebar.number_input('Total Funding (USD)', min_value=0, value=1000000)
    funding_rounds = st.sidebar.slider('Funding Rounds', 0, 20, 3)
    milestones = st.sidebar.slider('Milestones', 0, 10, 1)
    has_VC = st.sidebar.selectbox('Has VC Funding?', ['No', 'Yes'])
    has_angel = st.sidebar.selectbox('Has Angel Investment?', ['No', 'Yes'])
    has_roundA = st.sidebar.selectbox('Has Series A Round?', ['No', 'Yes'])
    has_roundB = st.sidebar.selectbox('Has Series B Round?', ['No', 'Yes'])
    has_roundC = st.sidebar.selectbox('Has Series C Round?', ['No', 'Yes'])
    has_roundD = st.sidebar.selectbox('Has Series D Round?', ['No', 'Yes'])
    avg_participants = st.sidebar.slider('Average Participants in Funding Rounds', 0, 50, 10)
    is_top500 = st.sidebar.selectbox('Is Top 500 Startup?', ['No', 'Yes'])

    city_enc = st.sidebar.number_input('City Code (encoded)', min_value=0, value=0)
    category_code_enc = st.sidebar.number_input('Category Code (encoded)', min_value=0, value=0)
    state_code_enc = st.sidebar.number_input('State Code (encoded)', min_value=0, value=0)

    return {
        'funding_total_usd': funding_total_usd,
        'funding_rounds': funding_rounds,
        'milestones': milestones,
        'has_VC': 1 if has_VC == 'Yes' else 0,
        'has_angel': 1 if has_angel == 'Yes' else 0,
        'has_roundA': 1 if has_roundA == 'Yes' else 0,
        'has_roundB': 1 if has_roundB == 'Yes' else 0,
        'has_roundC': 1 if has_roundC == 'Yes' else 0,
        'has_roundD': 1 if has_roundD == 'Yes' else 0,
        'avg_participants': avg_participants,
        'is_top500': 1 if is_top500 == 'Yes' else 0,
        'city_enc': city_enc,
        'category_code_enc': category_code_enc,
        'state_code_enc': state_code_enc,
    }

input_data = user_input_features()

if st.sidebar.button("Get Prediction"):
    # Convert input dict to dataframe for prediction
    input_df = pd.DataFrame([input_data])

    # Prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"The startup is predicted to SUCCEED 🎉 with probability {prediction_proba:.2f}")
    else:
        st.error(f"The startup is predicted to FAIL with probability {1 - prediction_proba:.2f}")

    st.write(f"Model accuracy on test data: {acc:.2f}")

    # Feature Importance plot
    st.subheader("Feature Importance")
    feat_importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    feat_importances.plot(kind='barh', ax=ax)
    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("Features")
    st.pyplot(fig)
