import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="GDP Predictor", layout="wide")

st.title("🌍 Economic Growth Predictor")

# DEBUG (check files)

st.write("FILES:", os.listdir())

# CHECK MODEL

if "model.pkl" not in os.listdir():
    st.error("model.pkl NOT FOUND ❌")
    st.stop()

model = pickle.load(open("model.pkl","rb"))

# LOAD DATA

df = pd.read_csv("final_structured_dataset.csv")
recent_df = pd.read_csv("final_structured_dataset_last5years.csv")

# TABS

tab1, tab2 = st.tabs(["Analysis","Prediction"])

# ---------------- ANALYSIS ----------------

with tab1:
    country = st.selectbox("Select Country", df['Country Name'].unique())

    filtered = df[df['Country Name']==country]
    recent = recent_df[recent_df['Country Name']==country]

    st.subheader("GDP Trend")
    st.line_chart(filtered.set_index('Year')['GDP'])

    st.subheader("Last 5 Years GDP")
    st.line_chart(recent.set_index('Year')['GDP'])

# ---------------- PREDICTION ----------------

with tab2:
    st.subheader("Predict GDP")

    inflation = st.number_input("Inflation")
    unemployment = st.number_input("Unemployment")
    life_exp = st.number_input("Life Expectancy")
    education = st.number_input("Education")
    gov = st.number_input("Government Spending")
    investment = st.number_input("Investment")
    trade = st.number_input("Trade")
    pop = st.number_input("Population Growth")

    if st.button("Predict"):
        data = np.array([[inflation, unemployment, life_exp, education, gov, investment, trade, pop]])
        pred = model.predict(data)

        st.success(f"Predicted GDP: {pred[0]:.2f}")
