from pydoc import doc
from flask import Flask, render_template, request
import tensorflow as tf
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
import h5py
from keras.applications.resnet50 import ResNet50

# Load the model

app = Flask(__name__)
model = ResNet50()

@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")


def teachable_machine_classification(img, breastcancerpred):
    model = load_model('breastcancerpred.h5')
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    print(prediction)
    return np.argmax(prediction) # return position of the highest probability


def predict(img):
    uploaded_file =img
    x=None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded image', use_column_width=True)
        label = teachable_machine_classification(image, 'breastcancerpred.h5')
        benign="The image is most likely benign"
        malignant="The image is most likely malignant"
        if label == 0:
          x=  ("The image is most likely 'Benign'")
        else:
          x=  ("The image is most likely 'Malignant'")
    return x
 

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['images']

		img_pa = "static/" + img.filename	
		img.save(img_pa)
    

		p = predict(img)               
      
	return render_template("index.html", prediction = p, img_path=img_pa)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
