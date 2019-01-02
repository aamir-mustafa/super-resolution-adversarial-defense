#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 09:09:48 2018

@author: aamir
"""

import matplotlib.pyplot as plt
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.measure import compare_psnr
import scipy.io as sio
import numpy as np
import os


n=5000 # number of images
path='adversarial_images' # path of adversarial images
adv_up=np.zeros((n,299,299,3))
label_file= 'Labels.mat' # true labels for adversarial images
feats=sio.loadmat(label_file)
labels=feats['labels']    
labels=labels.reshape((n))
count=0
names=os.listdir(path)
denoised_images_folder='denoised_images' #folder to save denoised images
os.mkdir(denoised_images_folder)

for i in range(n):
    ImgID= names(i)
    img = image.load_img(path +'/'+ ImgID ,target_size=(299, 299))
    image_rgb= image.img_to_array(img)
    
    im_bayes = denoise_wavelet(image_rgb/255, multichannel=True, convert2ycbcr=True,
                               method='BayesShrink', mode='soft',sigma=0.04)

    plt.imsave(denoised_images_folder+'/'+ ImgID+ '_wd.jpg', im_bayes)





