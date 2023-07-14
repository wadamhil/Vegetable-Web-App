# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 18:07:28 2023

@author: Adam Hilman
"""
import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import tensorflow_hub as hub



img_favicon = Image.open('icons/vegetable.png')

st.set_page_config(layout='wide',page_title="Veggies Detector", page_icon = img_favicon)
st.set_option('deprecation.showPyplotGlobalUse', False)

#st.beta_set_page_config(page_title='your_title', page_icon = favicon, layout = 'wide', initial_sidebar_state = 'auto')
# favicon being an object of the same kind as the one you should provide st.image() with (ie. a PIL array for example) or a string (url or local file path)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 



class KerasLayer(hub.KerasLayer):
    pass

# Load the model with custom object scope
with tf.keras.utils.custom_object_scope({'KerasLayer': KerasLayer}):
    loaded_model = tf.keras.models.load_model('resnetfinal.h5')




IMAGE_SIZE=224
filename= 'resnetfinal.h5'

CLASS_NAMES= ['Bean',
 'Bitter_Gourd',
 'Bottle_Gourd',
 'Brinjal',
 'Broccoli',
 'Cabbage',
 'Capsicum',
 'Carrot',
 'Cauliflower',
 'Cucumber',
 'Potato',
 'Radish',
 'Tomato']


img_banner = Image.open('icons/banner.jpg')

st.sidebar.image(img_banner)


with st.sidebar:
    i_page = st.selectbox('Vegetable Detection Using Images', ['Detector'], index=0)
    st.markdown("##### Developed by: Adam Hilman")


# Define the data augmentation parameters
data_augmentation = ImageDataGenerator(rescale=1/255.,
                                      rotation_range=40,
                                      shear_range=.2,
                                      zoom_range=.2,
                                      width_shift_range=.2,
                                      height_shift_range=.2,
                                      horizontal_flip=True)
    


if i_page== 'Detector':
    st.markdown("## Introduction")
    st.write(''' 
             The objective of this app is to perform real time detection of vegetables.
             User can upload the picture of the vegetable or click the picture via their camera.
             
            
             ''')
    st.markdown("---")        
    # load model
    #loaded_model= load_model(filename)
    # Load the model
    #loaded_model = tf.keras.models.load_model('resnetfinal.h5', custom_objects={'KerasLayer': KerasLayer})
    
    
    tab1, tab2= st.tabs(['Upload Image', 'Open Camera'])
    with tab1:
        img_file_buffer = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
    
        if img_file_buffer is not None:
            image = Image.open(img_file_buffer)
            st.image(image, width=200)

            image.save("Unseen/test.jpg")

            image = tf.keras.preprocessing.image.load_img(
                path=r'Unseen/test.jpg',
                target_size=(IMAGE_SIZE, IMAGE_SIZE),
            )

            # Apply data augmentation
            img_array = tf.keras.preprocessing.image.img_to_array(image)
            img_array = tf.expand_dims(img_array, 0)
            augmented_images = data_augmentation.flow(img_array, batch_size=1)

            # Predict on the augmented images
            for augmented_image in augmented_images:
                prediction = loaded_model.predict(augmented_image)
                prediction = prediction[0]
                index = np.argmax(prediction)
                predicted_label = CLASS_NAMES[index]
                predicted_confidence = prediction[index]
                st.subheader("This is probably a " +  CLASS_NAMES[index])
                st.markdown("##### Confidence: " + str(predicted_confidence*100) + " %")
                break  # Predict only on the first augmented image
        
    
    with tab2:
        cam_file_buffer = st.camera_input("Take a picture")
        if cam_file_buffer is not None:
                img = Image.open(cam_file_buffer)
                img.save("Unseen/cam.jpg")

                cam_image = tf.keras.preprocessing.image.load_img(
                    path=r'Unseen/cam.jpg',
                    target_size=(IMAGE_SIZE, IMAGE_SIZE),
                )

                # Apply data augmentation
                img_array = tf.keras.preprocessing.image.img_to_array(cam_image)
                img_array = tf.expand_dims(img_array, 0)
                augmented_images = data_augmentation.flow(img_array, batch_size=1)

                # Predict on the augmented images
                for augmented_image in augmented_images:
                    prediction = loaded_model.predict(augmented_image)
                    prediction = prediction[0]
                    index = np.argmax(prediction)
                    predicted_label = CLASS_NAMES[index]
                    predicted_confidence = prediction[index]
                    st.subheader("Probably a " +  CLASS_NAMES[index])
                    st.markdown("##### Confidence: " + str(predicted_confidence*100) + " %")
                    break  # Predict only on the first augmented image










