import streamlit as st
from prediction import predict

st.title('Classifying Parkinson\'s disease')
st.markdown('Parkinson\'s is a neurodegenerative disease that causes motor loss')

st.markdown('Please take a picture from your phone\'s camera and upload it to the platform.')

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

col1, col2 = st.columns(2)
with col1:
     st.markdown('Webcam Input')
     picture = st.camera_input("Take a picture")
with col2:
     st.markdown('File Upload')
     picture = st.file_uploader('Choose a File')

if picture is not None:
     st.write(predict(picture))
