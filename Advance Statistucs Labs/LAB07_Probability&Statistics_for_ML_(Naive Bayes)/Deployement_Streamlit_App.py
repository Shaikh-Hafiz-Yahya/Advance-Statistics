# Complete Implementation (Code) 

# Required Libraries
import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score 

st.title("Student Performance Predictor") 
uploaded_file = st.file_uploader("Upload CSV", type=["csv"]) 
if uploaded_file: 

    df = pd.read_csv(r'C:\Users\Muhammad Yahya\Downloads\Advance Statistucs Labs\LAB07_Probability&Statistics_for_ML_(Naive Bayes)\student_data.csv') 
  
    # Create target 
    df["Result"] = df["Marks"].apply(lambda x: 1 if x >= 60 else 0) 
 
    st.write(df.head()) 
 
    # Statistics 
    st.subheader("Statistical Report") 
 
    st.write("Mean:", df["Marks"].mean()) 
    st.write("Median:", df["Marks"].median()) 
    st.write("Variance:", df["Marks"].var()) 
    st.write("Std Dev:", df["Marks"].std()) 
 
    # Probability 
    pass_prob = len(df[df["Result"] == 1]) / len(df) 
    st.write("Probability of Passing:", pass_prob) 
 
    # Visualization 
    fig, ax = plt.subplots() 
    ax.hist(df["Marks"]) 
    st.pyplot(fig) 
 
    # ML Model 
    X = df[["Study_Hours", "Attendance", "Assignment"]] 
    y = df["Result"] 
 
    model = GaussianNB() 
    model.fit(X, y) 
 
    pred = model.predict(X) 
    acc = accuracy_score(y, pred) 
 
    st.write("Model Accuracy:", acc) 
 
    # Prediction UI 
    st.subheader("Predict") 
 
    study = st.slider("Study Hours", 0, 10) 
    att = st.slider("Attendance", 0, 100) 
    assign = st.slider("Assignment", 0, 100) 
 
    if st.button("Predict"): 
        result = model.predict([[study, att, assign]]) 
 
        if result[0] == 1: 
            st.success("PASS") 
        else: 
            st.error("FAIL") 
