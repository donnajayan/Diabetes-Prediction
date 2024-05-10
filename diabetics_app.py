import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

# Loading the saved model
loaded_model = pickle.load(open('trained_model.pkl', 'rb'))

def diabetics_prediction(input_data):
    # Changing the input_data to a NumPy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Load the scaler used during training
    sc = pickle.load(open('scaler.pkl', 'rb'))

    # Scale the input data using the same StandardScaler instance used for training
    input_data_scaled = sc.transform(input_data_reshaped)

    prediction = loaded_model.predict(input_data_scaled)

    if prediction == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

# Web page
def main():
    st.set_page_config(
        page_title='Diabetes Prediction App',
        page_icon="🏥"
    )

    st.title('Diabetes Prediction App')
    Pregnancies = st.text_input("Number of Pregnancies:")
    Glucose = st.text_input("Glucose Level:")
    BloodPressure = st.text_input("Blood Pressure Value:")
    SkinThickness = st.text_input("Skin Thickness Value:")
    Insulin = st.text_input("Insulin Value:")
    BMI = st.text_input("BMI Value:")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value:")
    Age = st.text_input("Age of the Person:")

    if st.button('Test Result'):
        if (
            Pregnancies != "" and
            Glucose != "" and
            BloodPressure != "" and
            SkinThickness != "" and
            Insulin != "" and
            BMI != "" and
            DiabetesPedigreeFunction != "" and
            Age != ""
        ):
            diagnosis = diabetics_prediction([
                float(Pregnancies),
                float(Glucose),
                float(BloodPressure),
                float(SkinThickness),
                float(Insulin),
                float(BMI),
                float(DiabetesPedigreeFunction),
                float(Age)
            ])
            st.success(diagnosis)
        else:
            st.warning("Please enter values for all the input features.")

if __name__ == '__main__':
    main()
