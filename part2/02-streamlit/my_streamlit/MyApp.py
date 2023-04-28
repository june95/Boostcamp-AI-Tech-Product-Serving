import streamlit as st

import io
import os
import yaml

from PIL import Image

from MyPredict import load_model, get_prediction

from My_confirm_button_hack import cache_on_button_press

st.set_page_config(layout="wide")

root_password = '1234'

def main():
    st.title("Mask Classification Model")

    with open("MyConfig.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    

    model = load_model()
    model.eval()

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg","png"])

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        
        st.image(image, caption='Uploaded Image')
        st.write("Classifying...")
        _, y_hat = get_prediction(model, image)
        label = config['classes'][y_hat.item()]

        st.balloons()
        st.write(f'label is {label}')


@cache_on_button_press('Authenticate')
def authenticate(password) ->bool:
    print(type(password))
    return password == root_password


password = st.text_input('password를 입력하시오', type="password")

if authenticate(password):
    st.success('You are authenticated!')
    main()
else:
    st.error('The password is invalid.')