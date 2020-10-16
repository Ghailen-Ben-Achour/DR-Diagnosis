# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 09:58:15 2020
@author: asus
"""
import os,random
import numpy as np
# charting
from PIL import Image
# metrics
from tensorflow.keras import backend as K
# keras
import tensorflow as tf
from tensorflow.keras.models import load_model

path='C:/Users/asus/Downloads/aptos2019-blindness-detection/train_images'
def representative_dataset_gen():
    for _ in range(20): 
        a=random.choice(os.listdir(path))
        file = path+'/'+a
        img=Image.open(file)
        img = img.resize((128,128))
        img = np.reshape(img,[1,128,128,3])
        yield [img]

model=load_model('C:/Users/asus/Desktop/model/3000neurons/trainable_brightnessweights_resnet3000.h5')
converter = tf.contib.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = tf.lite.RepresentativeDataset(
    representative_dataset_gen) 
tflite_quant_model = converter.convert()
open("activation_int.tflite","wb").write(tflite_quant_model)
print("convert model to tflite format done.")