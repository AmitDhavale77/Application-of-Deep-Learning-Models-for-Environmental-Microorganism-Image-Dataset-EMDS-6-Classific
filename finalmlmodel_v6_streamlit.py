#To explore mobilenetv2
#To explore VGG16
import os#all below imported packeages are important
import cv2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.callbacks import EarlyStopping
#%matplotlib inline
import matplotlib.pyplot as plt
from os import listdir
from PIL import Image 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from skimage.feature import hog
import json
from skimage import color
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score, precision_score,recall_score, f1_score 
import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
import os
from sklearn.model_selection import KFold
import streamlit as st
import pandas as pd # pip install pandas
from matplotlib import pyplot as plt # pip install matplotlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#Used to avoid printing of Warnings
st.title("EMDS6 model")
# get the path/directory
#folder_dir = "D:\\Learn_ML\\EMDS-6\\EMDS5-Original\\new"
#folder_dir1="D:\\Learn_ML\\EMDS-6\\EMDS5-Original\\01"
imagelist = []
filelist=[]
labellist=[]
featurelist=[]
fimagelist=[]
scaler = MinMaxScaler()

#streamlite commands

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('EMDS Image Classifier using Deep Learning')
st.text('Upload the Image from the listed category.\n[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]')

from keras.models import load_model
saved_model = load_model('best_model_mobilenetv2_final.hdf5')

# filename="EMDS5-g02-01.png"
# imge = Image.open(os.path.join(folder_dir,filename))
# imge.show()
# imge = np.array(imge)
# img1=cv2.resize(imge, (224, 224),interpolation = cv2.INTER_NEAREST)#The INTER_NEAREST method uses the nearest neighbor concept for interpolation. This is one of the simplest methods, using only one neighboring pixel from the image for interpolation.
# norm_image = cv2.normalize(img1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
# y = np.expand_dims(norm_image, axis=0)
# y_out = saved_model.predict(y)
# y_out=np.round(y_out)
# Categories = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21']    
# y_out1 = Categories[y_out.argmax()]
# print(y_out1)    

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","tiff","bmp"])
if uploaded_file is not None:
  imge = Image.open(uploaded_file)
  st.image(imge,caption='Uploaded Image')

  if st.button('PREDICT'):
    Categories = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21']    
    st.write('Result.....')
    flat_data=[]
    imge = np.array(imge)
    img1=cv2.resize(imge, (224, 224),interpolation = cv2.INTER_NEAREST)#The INTER_NEAREST method uses the nearest neighbor concept for interpolation. This is one of the simplest methods, using only one neighboring pixel from the image for interpolation.
    norm_image = cv2.normalize(img1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    y = np.expand_dims(norm_image, axis=0)
    y_out = saved_model.predict(y)
    y_out=np.round(y_out)
    Categories = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21']
    y_out1 = Categories[y_out.argmax()]
    st.title(f' PREDICTED OUTPUT: {y_out1}')
    q = saved_model.predict_proba(y)
    for index, item in enumerate(Categories):
      st.write(f'{item} : {q[0][index]*100}%')

st.text("")
st.text('Made by Amit Dhavale')

