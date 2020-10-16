# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 09:00:59 2020
@author: asus
"""




from tensorflow.keras import backend as K

import tensorflow as tf

from tensorflow.keras.models import load_model

path='C:/Users/asus/Downloads/aptos2019-blindness-detection/train_images'


   
model=load_model('C:/Users/asus/Desktop/model/3000neurons/trainable_brightnessweights_resnet3000.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.post_training_quantize = True

converter.optimizations = [tf.lite.Optimize.DEFAULT]


converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tflite_quant_model = converter.convert()
open("converted_model.tflite","wb").write(tflite_quant_model)
print("convert model to tflite format done.")