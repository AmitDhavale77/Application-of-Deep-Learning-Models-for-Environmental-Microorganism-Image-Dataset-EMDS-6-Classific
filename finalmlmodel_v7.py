#To explore mobilenetv2
#To explore VGG16
import os#all below imported packeages are important
import cv2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import Xception
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
from sklearn.metrics import classification_report,accuracy_score
import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# get the path/directory
folder_dir = "D:\\Learn_ML\\EMDS-6\\EMDS5-Original\\new"
imagelist = []
filelist=[]
labellist=[]
featurelist=[]
fimagelist=[]
scaler = MinMaxScaler()

# sq1x1 = "squeeze1x1"
# exp1x1 = "expand1x1"
# exp3x3 = "expand3x3"
# relu = "relu_"
 
# def fire_module(x, fire_id, squeeze=16, expand=64):
#    s_id = 'fire' + str(fire_id) + '/'
#    x = keras.layers.convolutional.Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
#    x = keras.layers.Activation('relu', name=s_id + relu + sq1x1)(x)
 
#    left = keras.layers.convolutional.Convolution2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
#    left = keras.layers.Activation('relu', name=s_id + relu + exp1x1)(left)
 
#    right = keras.layers.convolutional.Convolution2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
#    right = keras.layers.Activation('relu', name=s_id + relu + exp3x3)(right)
 
#    x = keras.layers.concatenate([left, right], axis=3, name=s_id + 'concat')
#    return x


for filename in os.listdir(folder_dir):
 
    # check if the image ends with png
    if (filename.endswith(".png")):
        img = cv2.imread(os.path.join(folder_dir,filename))
        # img1 = Image.open(os.path.join(folder_dir,filename))
        # filename1=filename.replace("jpg","png")
        # img1.save(os.path.join(folder_dir,filename1))
        #cv2.imwrite(os.path.join(folder_dir,filename1), img)
        print(filename)
        if img is not None:
            
           # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img1=cv2.resize(img, (224, 224),interpolation = cv2.INTER_NEAREST)
            norm_image = cv2.normalize(img1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            # calculate the hog and return a visual representation.
           # micro_hog, micro_hog_img = hog(norm_image, pixels_per_cell=(10,10),cells_per_block=(4,4),orientations=8,visualize=True,block_norm='L2-Hys')

            #img2 = scaler.transform(img1)
            imagelist.append(norm_image)
            filelist.append(filename)
           # featurelist.append(micro_hog)
            #fimagelist.append(micro_hog_img)
            s=filename
            result = s.find('g')
           # print(result)
            label=s[result+1:result+3]
            labellist.append(label)
            print(label)
print(imagelist[1].shape)
#cv2.imshow('image', imagelist[1])
#cv2.waitKey(0)
#cv2.destroyAllWindows()#all 3 statements must be required while using imshow
#cv2.waitKey(1)
labellist1 = [int(i) for i in labellist]
y = np.array(labellist1)#convert list into array
#x=np.array(featurelist)
x=np.array(imagelist)
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.20,random_state=0,stratify=y)

def to_categorical1(y, num_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy
    '''
    y = np.asarray(y, dtype='int32')
    if not num_classes:
        num_classes = np.max(y)
    Y = np.zeros((len(y), num_classes))
    for i in range(len(y)):
        Y[i, y[i]-1] = 1.
    return Y


y_train1 = to_categorical1(y_train, num_classes=21)
y_test1 = to_categorical1(y_test, num_classes=21)
#if random_state4 then we will get same labels will be selected as test or train else every time we run the train_test_split() we will get different samples

# calculate the hog and return a visual representation.
#micro_hog, micro_hog_img = hog(imagelist[1], pixels_per_cell=(14,14),cells_per_block=(2, 2),orientations=9,visualize=True,block_norm='L2-Hys')

# fig, ax = plt.subplots(1,2)
# fig.set_size_inches(8,6)
# # remove ticks and their labels
# [a.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False) 
#     for a in ax]
 
# ax[0].imshow(imagelist[100], cmap='gray')
# ax[0].set_title('microbes')
# ax[1].imshow(fimagelist[100], cmap='gray')
# ax[1].set_title('hog')
# plt.show()

#svm training accuracy=0.45628
# Accuracy: 0.4523809523809524


#               precision    recall  f1-score   support

#            1       0.80      1.00      0.89         4
#            2       1.00      0.75      0.86         4
#            3       0.43      0.75      0.55         4
#            4       0.80      1.00      0.89         4
#            5       0.25      0.25      0.25         4
#            6       0.80      1.00      0.89         4
#            7       0.50      0.50      0.50         4
#            8       0.00      0.00      0.00         4
#            9       0.00      0.00      0.00         4
#           10       0.00      0.00      0.00         4
#           11       1.00      0.75      0.86         4
#           12       0.00      0.00      0.00         4
#           13       0.40      0.50      0.44         4
#           14       0.25      0.25      0.25         4
#           15       0.20      0.25      0.22         4
#           16       0.40      0.50      0.44         4
#           17       0.67      0.50      0.57         4
#           18       1.00      0.75      0.86         4
#           19       0.00      0.00      0.00         4
#           20       0.50      0.25      0.33         4
#           21       0.40      0.50      0.44         4

#     accuracy                           0.45        84
#    macro avg       0.45      0.45      0.44        84
# weighted avg       0.45      0.45      0.44        84
#clf = svm.SVC()
#hog_features = np.array(x_train)
#data_frame = np.hstack((hog_features,y_train))
# clf.fit(x_train,y_train)
# y_pred = clf.predict(x_test)
# print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
# print('\n')
# print(classification_report(y_test, y_pred))

#Testing VGG16
# model = Sequential()
# model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
# model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Flatten())
# model.add(Dense(units=4096,activation="relu"))
# model.add(Dense(units=4096,activation="relu"))
# model.add(Dense(units=21, activation="softmax"))
# opt = Adam(lr=0.001)
# model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
# model.summary()

#print("loading VGG16 with imagenet weights")
print("loading MobileNetV2 with imagenet weights")
## Loading VGG16 model
#base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
#base_model = Xception(input_shape=(224,224,3), include_top=False, weights='imagenet', pooling='avg') # Average pooling reduces output dimensions

base_model.trainable = False ## Not trainable weights
base_model.summary()


# Model: "vgg16"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# input_1 (InputLayer)         [(None, 224, 224, 3)]     0         
# _________________________________________________________________
# block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
# _________________________________________________________________
# block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
# _________________________________________________________________
# block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
# _________________________________________________________________
# block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
# _________________________________________________________________
# block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
# _________________________________________________________________
# block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
# _________________________________________________________________
# block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
# _________________________________________________________________
# block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
# _________________________________________________________________
# block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
# _________________________________________________________________
# block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
# _________________________________________________________________
# block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
# _________________________________________________________________
# block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
# _________________________________________________________________
# block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
# _________________________________________________________________
# block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
# _________________________________________________________________
# block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
# _________________________________________________________________
# block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
# _________________________________________________________________
# block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
# _________________________________________________________________
# block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
# =================================================================
# Total params: 14,714,688
# Trainable params: 0
# Non-trainable params: 14,714,688
# _________________________________________________________________

from tensorflow.keras import layers, models

flatten_layer = layers.Flatten()
dense_layer_1 = layers.Dense(150, activation='selu',kernel_initializer="lecun_normal")
bn1=layers.BatchNormalization()
dp1=layers.Dropout(rate=0.2)
dense_layer_2 = layers.Dense(150, activation='selu',kernel_initializer="lecun_normal")
bn2=layers.BatchNormalization()
dp2=layers.Dropout(rate=0.2)
dense_layer_3 = layers.Dense(150, activation='selu',kernel_initializer="lecun_normal")
bn3=layers.BatchNormalization()
dp3=layers.Dropout(rate=0.2)
dense_layer_4 = layers.Dense(150, activation='selu',kernel_initializer="lecun_normal")
bn4=layers.BatchNormalization()
dp4=layers.Dropout(rate=0.2)
prediction_layer = layers.Dense(21, activation='softmax')

#squeezenet
# img_input = keras.layers.Input(shape=(224,224,3))
# x = keras.layers.convolutional.Convolution2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(img_input)
# x = keras.layers.Activation('relu', name='relu_conv1')(x)
# x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)
 
# x = fire_module(x, fire_id=2, squeeze=16, expand=64)
# x = fire_module(x, fire_id=3, squeeze=16, expand=64)
# x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)
 
# x = fire_module(x, fire_id=4, squeeze=32, expand=128)
# x = fire_module(x, fire_id=5, squeeze=32, expand=128)
# x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)
 
# x = fire_module(x, fire_id=6, squeeze=48, expand=192)
# x = fire_module(x, fire_id=7, squeeze=48, expand=192)
# x = fire_module(x, fire_id=8, squeeze=64, expand=256)
# x = fire_module(x, fire_id=9, squeeze=64, expand=256)
# x = keras.layers.Dropout(0.5, name='drop9')(x)
 
# x = keras.layers.convolutional.Convolution2D(21, (1, 1), padding='valid', name='conv10')(x)
# x = keras.layers.Activation('relu', name='relu_conv10')(x)
# x = keras.layers.GlobalAveragePooling2D()(x)
# out = keras.layers.Activation('softmax', name='loss')(x)
 
# model = keras.models.Model(img_input, out, name='squeezenet')


model = models.Sequential([
    base_model,
    flatten_layer,
    dense_layer_1,
    bn1,
    dp1,
    dense_layer_2,
    bn2,
    dp2,
    dense_layer_3,
    bn3,
    dp3,
    dense_layer_4,
    bn4,
    dp4,
    prediction_layer
])

model.summary()

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# vgg16 (Functional)           (None, 7, 7, 512)         14714688  
# _________________________________________________________________
# flatten (Flatten)            (None, 25088)             0         
# _________________________________________________________________
# dense (Dense)                (None, 50)                1254450   
# _________________________________________________________________
# dense_1 (Dense)              (None, 30)                1530      
# _________________________________________________________________
# dense_2 (Dense)              (None, 21)                651       
# =================================================================
# Total params: 15,971,319
# Trainable params: 1,256,631
# Non-trainable params: 14,714,688
# _________________________________________________________________


# checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
# early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
# hist = model.fit_generator(steps_per_epoch=100,generator=traindata, validation_data= testdata, validation_steps=10,epochs=100,callbacks=[checkpoint,early])


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)


es = EarlyStopping(monitor='val_accuracy', mode='max', patience=3,  restore_best_weights=True)
#mc = ModelCheckpoint('D:/Learn_ML/EMDS-5-main/best_model_squeezenet.hdf5', monitor='val_loss', mode='min', verbose=1,save_best_only=True)

#mc = ModelCheckpoint('D:/Learn_ML/EMDS-5-main/best_model_vgg16.hdf5', monitor='val_loss', mode='min', verbose=1,save_best_only=True)
mc = ModelCheckpoint('D:/Learn_ML/EMDS-5-main/best_model_mobilenet.hdf5', monitor='val_loss', mode='min', verbose=1,save_best_only=True)
#mc = ModelCheckpoint('D:/Learn_ML/EMDS-5-main/best_model_Xception.hdf5', monitor='val_loss', mode='min', verbose=1,save_best_only=True)

#print("training/fine tuning mobilenet last layers for microbes  dataset")
print("training/fine tuning Xception last layers for microbes  dataset")

hist=model.fit(x_train, y_train1, epochs=15, validation_split=0.2, batch_size=32, callbacks=[es, mc])

from keras.models import load_model
saved_model = load_model('D:/Learn_ML/EMDS-5-main/best_model_mobilenet.hdf5')
#saved_model = load_model('D:/Learn_ML/EMDS-5-main/best_model_vgg16.hdf5')
#saved_model = load_model('D:/Learn_ML/EMDS-5-main/best_model_squeezenet.hdf5')
#saved_model = load_model('D:/Learn_ML/EMDS-5-main/best_model_Xception.hdf5')


plt.plot(hist.history["accuracy"])
plt.plot(hist.history['val_accuracy'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()
#to test result in test data
# from keras.preprocessing import image
# img = image.load_img("image.jpeg",target_size=(224,224))
# img = np.asarray(img)
# plt.imshow(img)
# img = np.expand_dims(img, axis=0)
# from keras.models import load_model
# saved_model = load_model("vgg16_1.h5")
# output = saved_model.predict(img)
# if output[0][0] > output[0][1]:
#     print("cat")
# else:
#     print('dog')

_, train_acc = saved_model.evaluate(x_train, y_train1, verbose=0)
_, test_acc = saved_model.evaluate(x_test, y_test1, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))


# plot training history
plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='test')
plt.legend()
plt.show()

y_pred=saved_model.predict(x_test)
y_pred1=np.round(y_pred)
from sklearn.metrics import confusion_matrix
#cm=confusion_matrix(y_test1, y_pred1)

temp=[]
for i in y_pred1:
    temp.append(i.argmax(axis=0))# to extract position where output is 1
temp=list(temp)

temp1=[]
for i in y_test1:
    temp1.append(i.argmax(axis=0))               
xx= np.arange(0,21)

cm = confusion_matrix(temp1, temp, labels=xx)
recall = np.diag(cm) / np.sum(cm, axis = 1)
precision = np.diag(cm) / np.sum(cm, axis = 0)
recall1=np.mean(recall)
precision1=np.mean(precision)
f1=(2*recall1*precision1)/(recall1+precision1)
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(temp, temp1)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
#precision = precision_score(temp, temp1, pos_label='positive',average='micro')
print('Precision: %f'% np.mean(precision))
# recall: tp / (tp + fn)
#recall = recall_score(temp, temp1,pos_label='positive',average='micro')
print('Recall: %f' % np.mean(recall))
# f1: 2 tp / (2 tp + fp + fn)
#f1 = f1_score(temp, temp1)
print('F1 score: %f' % f1)

class_report = classification_report(temp1, temp, labels=np.arange(0,21))

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score


target= xx

# set plot figure size
fig, c_ax = plt.subplots(1,1, figsize = (12, 8))

# function for scoring roc auc score for multi-class
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    for (idx, c_label) in enumerate(target):
        fpr, tpr, thresholds = roc_curve(y_test[:,idx].astype(int), y_pred[:,idx])
        c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
    c_ax.plot(fpr, fpr, 'b-', label = 'Random Guessing')
    return roc_auc_score(y_test, y_pred, average=average)


print('ROC AUC score:', multiclass_roc_auc_score(temp1,temp))

c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
plt.show()

fig, px = plt.subplots(figsize=(7.5, 7.5))
px.matshow(cm, cmap=plt.cm.YlOrRd, alpha=0.5)
for m in range(cm.shape[0]):
    for n in range(cm.shape[1]):
        px.text(x=m,y=n,s=cm[m, n], va='center', ha='center', size='xx-large')

# Sets the labels
plt.xlabel('Predictions', fontsize=16)
plt.ylabel('Actuals', fontsize=16)
plt.title('Confusion Matrix', fontsize=15)
plt.show()

# imports
import seaborn as sebrn
# Using Seaborn heatmap to create the plot
fx = sebrn.heatmap(cm, annot=True, cmap='turbo')

# labels the title and x, y axis of plot
fx.set_title('Plotting Confusion Matrix using Seaborn\n\n');
fx.set_xlabel('Predicted Values')
fx.set_ylabel('Actual Values ');

# # labels the boxes
# fx.xaxis.set_ticklabels(['False','True'])
# fx.yaxis.set_ticklabels(['False','True'])

plt.show()
