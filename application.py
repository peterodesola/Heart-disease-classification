import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image



model=joblib.load('final_model')

# Collect user input features into dataframe
def features(input_data):
    input_data_as_numpy_array=np.asarray(input_data)
    input_reshape = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_reshape)

    if prediction[0] == 0:
        st.success('No trace of disease')

    else:
        return st.error('Warning!!! You need to see your doctor, you have high risk of having heart disease')

def main():
    st.title("HEART DISEASE PREDICTING SYSTEM USING MACHINE LEARNING")

    age = st.select_slider('Age', range(1,121,1))
    sex = st.radio('Sex Gender(Female=0,Male=1)',(0,1))
    cp = st.selectbox('Chest pain type(Typical Angina=0,Atypical Angina=1,Non-Angina=2,Asymptomatic=3)',(0,1,2,3))
    exang = st.radio('Exercise induced angina(No=0,Yes=1)',(0,1))
    oldpeak = st.number_input('oldpeak:ST Depression induced by exercise relative to rest (0-6)')
    slope = st.selectbox('The slope of the peak exercise ST segmen(Upsloping(betterHR)=0,Flatsloping(healthyheart)=1),Downsloping(unhealthy Heart)=2)', (0,1,2))
    ca = st.select_slider('(Ca)Number of major vessels colored by floroscopy',range(0,5,1))
    thal = st.select_slider('Thalium stress result',range(1,8,1))
    trestbps = st.text_input('Resting blood pressure')

    chol = st.text_input('Serum cholesterol in mg/dl')

    fbs = st.radio('Fasting blood sugar higher than 120mg/dl(No=0,Yes=1)', (0, 1))

    restecg = st.selectbox(
        'Resting electrocardiographic results(Nothing to note=0,ST-T wave abnormality=1,Possible left ventriclar hypertrophy=2)',
        (0, 1, 2))

    thalach = st.text_input('Thalach (Maximum heart rate achieved)')

    df = ''
    if st.button('Predict'):
        df = features([cp,thal,ca,exang,sex,slope,age,oldpeak,trestbps,chol,fbs,restecg,thalach])

if __name__ == '__main__':
    main()


st.sidebar.subheader("Information about the Application")

st.sidebar.info("This web application is designed to predict heart disease status.")
st.sidebar.info("You are to enter the required information and click on the 'Predict' button to check your heart status")
st.sidebar.info("The heart disease predicting system is designed using ensemble technique (RF, XGB AND DT)")
st.sidebar.info('This project study was carried out by Odesola Peter, under the supervision of Dr Hamidreza Soltani.' )


df = pd.read_csv("heart.csv")
st.sidebar.dataframe(df)


image=Image.open('heart.png')
st.image(image, caption='Heart Disease Prediction using ML', use_column_width=True)
