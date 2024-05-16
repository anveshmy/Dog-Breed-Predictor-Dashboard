import streamlit as st
from PIL import Image
from prediction import core_ml_image_prediction
import altair as alt

st.title("Dog Breed Predictor")


models = {
    "CoreML" : "models/Dog Breed Classification Model 1 copy.mlmodel"
    }

option = st.selectbox(
   "What model would you like to use?",
   ("CoreML", "PyTorch"),
   index=None,
   placeholder="Select model...",
)

if option == "CoreML":
    uploaded_image = st.file_uploader('Upload your Doggy', type=['png', 'jpg'])
    top_number = st.number_input("Insert number of Breed Probabilities to return", min_value = 1)
    if uploaded_image:
        st.image(uploaded_image)
        _ , _ , prediction = core_ml_image_prediction(models[option], uploaded_image)
        rounded_probabilities = {}
        #st.write(f"Prediction: {prediction['probabilities']}")
       # st.bar_chart(prediction['probabilities'], x = 'Breed', y='Probability', color ="#1C401C")
        st.write(alt.Chart(prediction['probabilities'].head(top_number)).mark_bar().encode(x=alt.X('Breed', sort=None), y='Probability'))
        st.dataframe(prediction['probabilities'].head(top_number))
elif option == "PyTorch":
    st.error("Model Coming Soon!")



if __name__ == "__main__":
    # This condition prevents the Streamlit app from running itself recursively
    if "__streamlitmagic__" not in locals():
        st.web.bootstrap.run(__file__, None, [], {})
