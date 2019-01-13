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
from keras.preprocessing import image

n=1 # number of images
path='adversarial_images' # path of adversarial images
names=os.listdir(path)
denoised_images_folder='test' #folder to save denoised images
sigma= [0.0, 0.01, 0.02, 0.03, 0.04] #chose a smaller sigma for small perturbation sizes
for i in range(n):
    ImgID= names[i].split('.')[0]
    img = image.load_img(path +'/'+ ImgID +'.png',target_size=(299, 299))
    image_rgb= image.img_to_array(img)
    
    im_bayes = denoise_wavelet(image_rgb/255, multichannel=True, convert2ycbcr=True,
                               method='BayesShrink', mode='soft',sigma=sigma[1])

    plt.imsave(denoised_images_folder+'/'+ ImgID+ '_wd.jpg', im_bayes)





