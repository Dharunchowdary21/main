import os
import sys
import base64
import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import colors
from PIL import Image

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background("bg.jpg")


st.markdown("<h1 style='text-align: center; color: black;'>Next day fire spread</h1>", unsafe_allow_html=True)


test_inputs = st.file_uploader('FILE UPLOAD')

if test_inputs is not None:

    test_inputs = np.load(test_inputs)
    model2 = tf.keras.models.load_model("load_model/next_day_fire_spread_predict.h5")

    y_pred = model2.predict(test_inputs)

    fig = plt.figure(figsize=(15,6.5))

    new_titles = TITLES = ['Elevation',
    'Wind\ndirection',
    'Wind\nvelocity',
    'Min\ntemp',
    'Max\ntemp',
    'Humidity',
    'Precip',
    'Drought',
    'Vegetation',
    'Population\ndensity',
    'Energy\nrelease\ncomponent',
    'Previous\nfire\nmask',
    'Fire\nmask'
    ]
    new_titles.append("Predicted fire\nmask")


    # Number of rows of data samples to plot
    n_rows = 5 
    # Number of data variables
    n_features = test_inputs.shape[3]
    # Variables for controllong the color map for the fire masks
    CMAP = colors.ListedColormap(['black', 'silver', 'orangered'])
    BOUNDS = [-1, -0.1, 0.001, 1]
    NORM = colors.BoundaryNorm(BOUNDS, CMAP.N)

    for i in range(n_rows):
        for j in range(n_features + 1):
            plt.subplot(n_rows, n_features + 1, i * (n_features + 1) + j + 1)
            if i == 0:
                plt.title(new_titles[j], fontsize=13)
            if j < n_features - 2:
                plt.imshow(test_inputs[i, :, :, j], cmap='viridis')
            if j == n_features - 2:
                plt.imshow(test_inputs[i, :, :, -1], cmap=CMAP, norm=NORM)
            if j == n_features -1 :
                plt.imshow(y_pred[i], cmap=CMAP, norm=NORM)
            if j == n_features : 
                plt.imshow(y_pred[i], cmap=CMAP, norm=NORM)
        plt.axis('off')
    plt.tight_layout()
    st.success("Successfully produces output.")
    st.pyplot(plt)

    quit()