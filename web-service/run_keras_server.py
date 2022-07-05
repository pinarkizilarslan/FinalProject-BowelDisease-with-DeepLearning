
from __future__ import division, print_function
from base64 import decode
from unittest import result
#from keras.preprocessing.image import  ImageDataGenerator, array_to_img, img_to_array
from keras.utils import img_to_array,load_img 
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
from flask import Flask, redirect,url_for,request,render_template, jsonify
import io
import base64
from sqlalchemy import subquery
from werkzeug.utils import secure_filename
import os
from matplotlib import image
import tensorflow as tf
import cv2
import math
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from models import app,Doctor,Patient,Images,db,Access
from PIL import Image
from io import BytesIO
import base64

UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def crop(img):

  blurred = cv2.blur(img, (3,3))
  canny = cv2.Canny(blurred, 50, 200)
  ## find the non-zero min-max coords of canny
  pts = np.argwhere(canny>0)
  y1,x1 = pts.min(axis=0)
  y2,x2 = pts.max(axis=0)
  ## crop the region
  cropped = img[y1:y2, x1:x2]

  return cropped

def convolution(image, kernel, average=False):

  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image_row, image_col = image.shape
  kernel_row, kernel_col = kernel.shape
  output = np.zeros(image.shape)

  pad_height = int((kernel_row - 1) / 2)
  pad_width = int((kernel_col - 1) / 2)
  padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
  padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image
  
  for row in range(image_row):
    for col in range(image_col):
      output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
      if average:
        output[row, col] /= kernel.shape[0] * kernel.shape[1]
 
  return output

def dnorm(x, mu, sd):

  return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)
 
def gaussian_kernel(size, sigma=1):

  kernel_1D = np.linspace(-(size // 2), size // 2, size)
  for i in range(size):
    kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)

  kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
  kernel_2D *= 1.0 / kernel_2D.max()
  
  return kernel_2D
 
 
def gaussian_blur(image, kernel_size):

  kernel = gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size))
  return convolution(image, kernel, average=True)
 
 
def histogram_eq(image):

  img_array = np.asarray(image)
  img_array = img_array.astype(int)
  histogram_array = np.bincount(img_array.flatten(), minlength=256)
    
  num_pixels = np.sum(histogram_array)
  histogram_array = histogram_array/num_pixels

  chistogram_array = np.cumsum(histogram_array)

  transform_map = np.floor(255 * chistogram_array).astype(np.uint8)

  img_list = list(img_array.flatten())

  eq_img_list = [transform_map[p] for p in img_list]
  eq_img_array = np.reshape(np.asarray(eq_img_list), img_array.shape)

  backtorgb = cv2.cvtColor(eq_img_array,cv2.COLOR_GRAY2RGB)
    
  return backtorgb

@app.route("/",methods=["GET","POST"])
def login():
  if request.method=="POST":
        email = request.form.get('email')
        password = request.form.get('password')

        user = Doctor.query.filter_by(email=email).first()

        if not user or user.password!=password:
          return redirect(url_for('login'))

        return redirect(url_for('index',id=user.id))  

  return render_template('login.html')


@app.route("/index/<id>",methods=["GET","POST"])
def index(id):

    user = Doctor.query.filter_by(id=id).first()

    if request.method == "POST":
      tcno = request.form.get('tcno')
      patient=Patient.query.filter_by(tcno=tcno).first()
      if not patient:
          print("Hasta Bulunamadı")
          return redirect(url_for('index',id=user.id))
      acc=Access.query.filter_by(doctor_id=user.id,patient_id=patient.id).first()
      if not acc:
         return redirect(url_for('index',id=user.id))

      diagnosis=Images.query.filter_by(patient_id=patient.id).first()
      if diagnosis:
        return render_template("index.html",patient=patient,user=user,diagnosis=diagnosis)

      return render_template("index.html",patient=patient,user=user)
        
    return render_template("index.html",user=user)


@app.route("/upload/<id>,<pid>",methods=["GET","POST"])
def upload(id,pid):
  user = Doctor.query.filter_by(id=id).first()
  patient = Patient.query.filter_by(id=pid).first()
  if request.method == "POST":
        print("tetiklendi")
        filestr = request.files['file'].read()
        npimg = np.fromstring(filestr, np.uint8)
        input = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        cropped = crop(input)
        gaussian = gaussian_blur(cropped, 5)
        histogram = histogram_eq(gaussian)

        input = cv2.resize(histogram,(224,224))
        input = input.reshape(1,224,224,3)

        modelDia = tf.keras.models.load_model("model_DIA_ckpt.h5")
        predsdia = modelDia.predict(input)
        classesdia = predsdia.argmax(axis=-1) 

        model = tf.keras.models.load_model("model_ckpt.h5")
        preds = model.predict(input)
        classes = preds.argmax(axis=-1)

        resultdia=" "

        if classesdia == [0]:
            resultdia = "Hastalıklı Bağırsak"

        elif classesdia == [1]:
            resultdia = "Sağlıklı Bağırsak"


        result = " "

        if classes == [0]:
            result = "CANCER"

        elif classes == [1]:
            result = "CROHNS"
        
        elif classes == [2]:
            result = "NORMAL"
        
        elif classes == [3]:
            result = "POLYP"

        elif classes==[4]:
            result = "ULCERATIVE COLITS"

       

        imageurl = base64.b64encode(filestr)

        newDiagnosis = Images(imageurl=imageurl,result=resultdia+"-"+result,patient_id=pid)
        db.session.add(newDiagnosis)
        db.session.commit()
        filestrlast = base64.b64decode(imageurl)
        return render_template("predict.html",result=resultdia+"-"+result,user=user, imageurl=request.files['file'])
  return render_template("predict.html",user=user)    




if __name__=='__main__':
    app.run(debug=True)