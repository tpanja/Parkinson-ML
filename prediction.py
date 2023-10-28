import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import streamlit as st
from filesplit.merge import Merge
import os

def transform_image(image):
     image = np.array(Image.open(image))

     image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
     
     edged = cv2.Canny(gray, 100,200)
          
     # Finding Contours
     # Use a copy of the image e.g. edged.copy()
     # since findContours alters the image
     contours, hierarchy = cv2.findContours(edged, 
     cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
     
     #print("Number of Contours found = " + str(len(contours)))
     # Draw all contours
     # -1 signifies drawing all contours
     
     mask = np.zeros([image.shape[0], image.shape[1], image.shape[2]], image.dtype)
     if len(contours) > 0:
          c = max(contours, key = cv2.contourArea)
          #print(c.shape)
          epsilon = 0.04 * cv2.arcLength(c, True)
          approx = cv2.approxPolyDP(c, epsilon, True)
          if len(approx) == 4:
               cv2.drawContours(image, [approx], -1, (0, 255, 0), 4)
               #x,y,w,h = cv2.boundingRect(c)
               corners = approx.reshape(-1, 2)
               for corner in corners:
                    x, y = corner
                    cv2.circle(image, (x, y), 8, (0, 0, 255), -1)
               
               pts = approx
               cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
               #dst = cv2.bitwise_and(image, image, mask=mask)
               # List the output points in the same order as input
               # Top-left, top-right, bottom-right, bottom-left
               width, height = image.shape[0], image.shape[1]
               dstPts = [[0, 0], [width, 0], [width, height], [0, height]]
               # Get the transform
               m = cv2.getPerspectiveTransform(np.float32(corners), np.float32(dstPts))
               # Transform the image
               out = cv2.warpPerspective(image_copy, m, (int(width), int(height)))
               dim = (224, 224)
               # Save the output
          #cv2.drawContours(image, [c], -1, (0, 255, 0), 3)

          #cv2.rectangle(image, (x, y), (x+w, y+h), (255,0,0), 3)

     print(image.shape)
     print(mask.shape)

     try:
          out = cv2.resize(out[20:-20, 20:-20], dim, interpolation = cv2.INTER_AREA)
     except Exception:
          out = np.zeros((224, 224, 3))
     
     st.image(image, caption='Extracted Spiral')
     return out

def predict(image):
     resized_image = transform_image(image)
     
     st.image(resized_image, caption='Output Image')

     resized_image = resized_image[np.newaxis, :, :, :]
     #st.write(os.getcwd())
     #os.chdir('/Users/tanaypanja/Documents/Parkinson-ML')
     if os.path.exists('model.h5'):
          print('Model already exists')
     else:
          merge = Merge('./separated_model', '.', 'model.h5')
          merge.merge()
          print('Model File Created')
     model = load_model('model.h5')

     layer = tf.keras.layers.Softmax()
     prediction = layer(model.predict(resized_image)[0]).numpy()[1]
     if prediction > 0.5:
          return 'You have Parkinson\'s disease with ' + str(int(prediction*100)) + '% confidence.'
     return 'You do not have Parkinson\'s disease.'

