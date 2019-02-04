from flask import Flask, render_template,request
#scientific computing library for saving, reading, and resizing images
from scipy.misc.pilutil import imsave, imread, imresize
#for matrix math
import numpy as np
#for importing our keras model
import keras.models
#for regular expressions, saves time dealing with string data
import base64
import re
import cv2 as cv
import sys
import os
from os import walk, getcwd

sys.path.append(os.path.abspath("./model"))
from model.load import *

app = Flask(__name__)
global model

model, graph = init()


mypath = "data/"
txt_name_list = []
for (dirpath, dirnames, filenames) in walk(mypath):
    if filenames != '.DS_Store':
        txt_name_list.extend(filenames)
        break

for ndx, member in enumerate(txt_name_list):
    txt_name_list[ndx] = txt_name_list[ndx].replace('.npy', '')


def convertImage(imgData1):
    imgstr = re.search(b'base64,(.*)',imgData1).group(1)
    with open('output.png','wb') as output:
      output.write(base64.b64decode(imgstr))


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    imgData = request.get_data()
    convertImage(imgData)
    print("debug")

    x = imread('output.png', mode='L')

    x = preprocess(x)

    x = imresize(x, (28, 28))

    x = x.astype('float32')
    x /= 255

    x = x.reshape(1, 28, 28, 1)
    print("debug2")

    with graph.as_default():

        out = model.predict(x)
        #out = model.predict_proba(x, verbose=1)
        print(out)
        print(np.argmax(out, axis=1))
        index = np.array(np.argmax(out, axis=1))
        index = index[0]
        sketch = txt_name_list[index]
        print("debug3")
        return sketch

@app.route('/getNoteText',methods=['GET','POST'])
def GetNoteText():
    if request.method == 'POST':
        file = request.files['pic']
        filename = file.filename
        #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #processImage(filename)
        return "Success"
    else:
        return "Y U NO USE POST?"


def adjust_gamma(image, gamma=1.5):
   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv.LUT(image, table)


def preprocess(img):
    # for sketch & not canvas drawings use the following:

    # gray = cv.bilateralFilter(img, 9, 75, 75)
    #
    # gray = cv.erode(gray, None, iterations=1)
    #
    # gray = adjust_gamma(gray, 1.1)

    th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv.THRESH_BINARY_INV, 11, 2)
    return th3

if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port)
