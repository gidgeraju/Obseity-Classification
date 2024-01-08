# Obseity-Classification
Created an innovative Obesity Checker utilizing a machine learning algorithm.  Designed the system to analyze user-provided health metrics, employing a custom-tailored  algorithm for accurate obesity classification. The algorithm, trained on a diverse dataset, exhibits  the ability to provide real-time predictions.


python code:

import streamlit as st
import joblib
from PIL import Image

model = joblib.load("LRoc.pkl")

bmi_labels = ["Normal Weight", "Obese", "Overweight", "Underweight"]

def predict_bmi_category(age, gender, height, weight):
    bmi = weight / (height / 100) ** 2
    bmi_label = model.predict([[age, gender, height, weight]])[0]
    bmi_category = bmi_labels[int(bmi_label)]
    return bmi_label, bmi_category, bmi


st.set_page_config(page_title="Obesity Prediction & Classification - MPML19", page_icon="🩺")
st.title("Machine Learning Approach for the Prediction of Obesity")

image = Image.open("bgimg.jpg")
st.image(image, caption='Obesity Prediction & Classification', use_column_width=True)

st.write("Enter your information:")

age = st.number_input("Age", min_value=1, max_value=100, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
height = st.number_input("Height (in cm)", min_value=0, max_value=300, step=1)
weight = st.number_input("Weight (in kg)", min_value=0, max_value=500, step=1)

if st.button("Predict Obesity", key="predict_button"):
    gender = 1 if gender == "Male" else 0
    bmi_label, bmi_category, bmi_value = predict_bmi_category(age, gender, height, weight)

    st.markdown(f"**BMI (Body Mass Index):** {bmi_value:.2f}")
    st.markdown(f"**Label:** {bmi_label}")
    st.markdown(f"**Output:** {bmi_category}")

st.markdown("---")  

st.write("### About:")
st.write("")

if st.button("Learn More", key="learn_more_button", help="Click to learn more about obesity"):
    st.write("Obesity is a complex, chronic disease with several causes that lead to excessive body fat and sometimes, poor health." 
             "Body fat itself is not a disease, of course. But when your body has too much extra fat, it can change the way it functions."
              "These changes are progressive, can worsen over time, and they can lead to adverse health effects."
                "The good news is that you can improve your health risks by losing some of your excess body fat. Even small changes in weight can have a big impact on your health." 
                "Not every weight loss method works for everyone. Most people have tried to lose weight more than once. And keeping the weight off is just as important as losing it in the first place.")
    st.write("BMI Categories:")
    st.write("Underweight: BMI < 18.5")
    st.write("Normal Weight: 18.5 <= BMI < 24.9")
    st.write("Overweight: 25 <= BMI < 29.9")
    st.write("Obese: BMI >= 30")
