"""
Created on Tue Dec  4 15:00:32 2018

@author: aamir
"""

#Here Inception v3 is used as a classifier.
import os
import numpy as np
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
imodel= InceptionV3(include_top=True, weights='imagenet')
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as sio


#Sample Clean Images
clean_folder='clean_images'
n=1 # number of images
loaded_clean=np.zeros((1,299,299,3))
names=os.listdir(clean_folder)

for i in range(n):
    name=names[i].split('.')[0]
    ImgID= name+'.png'
    img = image.load_img(clean_folder +'/'+ ImgID ,target_size=(299, 299))
    image_rgb= image.img_to_array(img)
    loaded_clean[i]=image_rgb 


pred_clean = imodel.predict(preprocess_input(loaded_clean))
print("Top 5 predictions (: ", decode_predictions(pred_clean, top=3))


#Sample Adversarial Images
adv_folder='adversarial_images'

loaded_adv=np.zeros((n,299,299,3))
names=os.listdir(adv_folder)

for i in range(n):
    name=names[i].split('.')[0]
    ImgID= name+'.png'
    img = image.load_img(adv_folder +'/'+ ImgID ,target_size=(299, 299))
    image_rgb= image.img_to_array(img)
    loaded_adv[i]=image_rgb 

pred_adv = imodel.predict(preprocess_input(loaded_adv))
print("Top 5 predictions (: ", decode_predictions(pred_adv, top=3))


#Recovered Images
recovered_folder='experiment/test/results-Demo'

loaded_recovered=np.zeros((n,598,598,3))

for i in range(n):
    name=names[i].split('.')[0]
    ImgID= name+'_wd_x2_SR.png'
    img = image.load_img(recovered_folder +'/'+ ImgID ,target_size=(598, 598))
    image_rgb= image.img_to_array(img)
    loaded_recovered[i]=image_rgb 

pred_rec = imodel.predict(preprocess_input(loaded_recovered))
print("Top 5 predictions (: ", decode_predictions(pred_rec, top=3))









