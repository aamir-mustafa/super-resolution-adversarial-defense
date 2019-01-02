"""
Created on Tue Dec  4 15:00:32 2018

@author: aamir
"""

#Here Inception v3 is used as a classifier.

import numpy as np
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
imodel= InceptionV3(include_top=True, weights='imagenet')
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as sio

adv_folder='adversarial_images'
n=5000 # number of images
loaded_adv=np.zeros((n,299,299,3))
label_file= 'Labels.mat' # true labels for adversarial images
feats=sio.loadmat(label_file)
labels=feats['labels']    
labels=labels.reshape((n))
names=os.listdir(adv_folder)

for i in range(n):
    name=names[i].split('.')[0]
    ImgID= name+'.png'
    img = image.load_img(adv_folder +'/'+ ImgID ,target_size=(299, 299))
    image_rgb= image.img_to_array(img)
    loaded_adv[i]=image_rgb 


pred_adv = imodel.predict(preprocess_input(loaded_adv))
pred_labels_top1_integer_adv=np.argmax(pred_adv, axis=1) 
a=np.where(pred_labels_top1_integer_adv==labels)
print ('Top-1 Accuracy on Adv Images is: ', len(a[0])/n*100, ' %')


recovered_folder='experiment/test/results_Demo'
n=5000 # number of images
loaded_recovered=np.zeros((n,598,598,3))

for i in range(n):
    name=names[i].split('.')[0]
    ImgID= name+'_x2_SR.png'
    img = image.load_img(recovered_folder +'/'+ ImgID ,target_size=(598, 598))
    image_rgb= image.img_to_array(img)
    loaded_recovered[i]=image_rgb 


pred_adv = imodel.predict(preprocess_input(loaded_recovered))
pred_labels_top1_integer_adv=np.argmax(pred_adv, axis=1) 
a=np.where(pred_labels_top1_integer_adv==labels)
print ('Top-1 Accuracy on Recovered Images is: ', len(a[0])/n*100, ' %')








