# -*- coding: utf-8 -*-
"""For Deployment Solar_Panel_Regression_Group_4.ipynb"""

import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
import streamlit as st

# =========================================================
#  LOAD CSV AND TRAIN MODEL
# =========================================================

df = pd.read_csv('solarpowergeneration.csv')

# Fill missing values
df['average-wind-speed-(period)'] = df['average-wind-speed-(period)'].fillna(
    df['average-wind-speed-(period)'].mean()
)

# Separate features & target
df_features = df.drop(['power-generated'], axis=1)
y = df['power-generated']

# =========================================================
#  SCALERS: ROBUST + STANDARD
# =========================================================

robust_cols = ["wind-direction", "visibility"]
standard_cols = [col for col in df_features.columns if col not in robust_cols]

robust_scaler = RobustScaler()
standard_scaler = StandardScaler()

# Fit scalers
df_features_robust = pd.DataFrame(
    robust_scaler.fit_transform(df_features[robust_cols]),
    columns=robust_cols
)

df_features_standard = pd.DataFrame(
    standard_scaler.fit_transform(df_features[standard_cols]),
    columns=standard_cols
)

# Combine
scaled_df = pd.concat([df_features_standard, df_features_robust], axis=1)

# Save scaler objects
with open("scalers.pkl", "wb") as f:
    pickle.dump((robust_scaler, standard_scaler, robust_cols, standard_cols), f)


# =========================================================
#  MANUAL SPLIT (80/20)
# =========================================================

np.random.seed(42)
n_samples = len(scaled_df)
shuffled_indices = np.random.permutation(n_samples)
train_size = int(0.8 * n_samples)

train_indices = shuffled_indices[:train_size]
test_indices = shuffled_indices[train_size:]

x_train = scaled_df.iloc[train_indices]
x_test = scaled_df.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

# =========================================================
#  TRAIN MODEL (UPDATED PARAMS)
# =========================================================

model = GradientBoostingRegressor(
    subsample=0.8,
    n_estimators=100,
    min_samples_split=5,
    min_samples_leaf=4,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)

model.fit(x_train, y_train)

# Save model
with open('gradient_boosting_model.pkl', 'wb') as f:
    pickle.dump(model, f)


# =========================================================
# STREAMLIT UI
# =========================================================

st.set_page_config(
    page_title="Solar Panel Regression App",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
    <style>
        .main { background-color: #f4f7fa; }
        .title-text { font-size: 40px; font-weight: 800; color: #1B7F79; text-align: center; padding-bottom: 10px; }
        .sub-text { text-align: center; font-size: 20px; color: #333; margin-top: -15px; padding-bottom: 20px; }
        .input-card { background: white; padding: 25px; border-radius: 15px; box-shadow: 0px 4px 12px rgba(0,0,0,0.1); }
        .prediction-box { background: #1B7F79; color: white; padding: 18px; border-radius: 12px; 
                          text-align: center; font-size: 24px; font-weight: 700; margin-top: 20px; }
        .footer-text { text-align:center; margin-top:40px; color:#555; font-size:18px; }
    </style>
""", unsafe_allow_html=True)


# Header
st.markdown("<h1 class='title-text'>⚡ Solar Panel Regression App</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Gradient Boosting Power Generation Prediction</p>", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL + SCALERS
# =========================================================

@st.cache_resource
def load_artifacts():
    with open("gradient_boosting_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scalers.pkl", "rb") as f:
        robust_scaler, standard_scaler, robust_cols, standard_cols = pickle.load(f)

    return model, robust_scaler, standard_scaler, robust_cols, standard_cols


model, robust_scaler, standard_scaler, robust_cols, standard_cols = load_artifacts()


# =========================================================
#  USER INPUT UI
# =========================================================

st.markdown("<div class='input-card'>", unsafe_allow_html=True)
st.markdown("### 🌤 Enter Environmental Parameters")

cols = st.columns(3)
user_input = {}

for i, col in enumerate(df_features.columns):
    with cols[i % 3]:
        user_input[col] = st.number_input(col.replace("-", " ").title(), value=float(df_features[col].mean()))

st.markdown("</div>", unsafe_allow_html=True)

user_df = pd.DataFrame([user_input])


# =========================================================
#  APPLY SCALING BASED ON RULES
# =========================================================

def apply_scaling(df):
    robust_scaled = pd.DataFrame(
        robust_scaler.transform(df[robust_cols]),
        columns=robust_cols
    )

    standard_scaled = pd.DataFrame(
        standard_scaler.transform(df[standard_cols]),
        columns=standard_cols
    )

    return pd.concat([standard_scaled, robust_scaled], axis=1)


scaled_user_df = apply_scaling(user_df)


# =========================================================
# PREDICTION
# =========================================================

if st.button("🔍 Predict Power Generation", use_container_width=True):

    # Rule: if all inputs are zero → prediction = 0
    if np.allclose(user_df.to_numpy().flatten(), 0.0):
        st.info("All inputs are zero — predicted power = 0 kW")
        st.markdown("<div class='prediction-box'>🌞 Predicted Power: <br>0.00 kW</div>", unsafe_allow_html=True)
    else:
        prediction = model.predict(scaled_user_df)[0]
        st.markdown(
            f"<div class='prediction-box'>🌞 Predicted Power: <br>{prediction:.2f} kW</div>",
            unsafe_allow_html=True
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("<p class='footer-text'>App Created by <b>Manish Maltare</b></p>", unsafe_allow_html=True)
