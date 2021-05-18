#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Network

# In[ ]:


get_ipython().system('nvidia-smi')


# ### Importing the libraries

# In[ ]:


import tensorflow as tf
import cv2
import glob
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, InceptionResNetV2, MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import numpy as np
import os


# In[ ]:


tf.__version__


# ## Part 1 - Data Preprocessing

# 
# ### Preprocessing the Training set

# In[ ]:


#Read images and convert them to arrays
train_images = []
train_lables = []
for directory_path in glob.glob("../input/sbir151/data/train/*"):
    lables = directory_path.split("/")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        train_images.append(img)
        train_lables.append(lables)
train_images = np.array(train_images)
train_lables = np.array(train_lables)


# 
# 
# ### Preprocessing the Test set

# In[ ]:


#Read images and convert them to arrays
test_images = []
test_lables = []
for directory_path in glob.glob("../input/sbir151/data/val/*"):
    lables = directory_path.split("/")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        test_images.append(img)
        test_lables.append(lables)
test_images = np.array(test_images)
test_lables = np.array(test_lables)


# In[ ]:


#Creating an encoding for the class labels
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(test_lables)
test_lables_en = le.transform(test_lables)
le.fit(train_lables)
train_lables_en = le.transform(train_lables)


# In[ ]:


#Assign training, testing images and the labels
x_train,y_train,x_test,y_test = train_images,train_lables_en,test_images,test_lables_en


# In[ ]:


#Model to deduce the features
model = VGG16 (weights='imagenet',include_top=False,input_shape=(100,100,3))

for layer in model.layers:
    layer.trainable = True


# In[ ]:


#Obtain the features which will be reduces numpy arrays
import sys
import numpy
feature_extractor = model.predict(x_train)
print(feature_extractor)
shape =feature_extractor.shape
print(shape)


# In[ ]:


#reshaping to 2d array
feature = feature_extractor.reshape(feature_extractor.shape[0],-1)
print(feature)


# In[ ]:


#Using SVM to train on the features extracted using the pretrained architecture
from sklearn import svm
svmmodel = svm.SVC(kernel='linear',verbose=True)
svmmodel.fit(feature,y_train)
svmmodel.score(feature,y_train)


# In[ ]:


#Predicting with the classifier model
predicted_train = svmmodel.predict(feature)
predicted_train = le.inverse_transform(predicted_train)


# In[ ]:


import pickle
s = pickle.dump(svmmodel,open('../working/svm_vgg_16.h5', 'wb'))
print(predicted_train)
print(le.inverse_transform(y_train))


# In[ ]:



from sklearn.metrics import accuracy_score,confusion_matrix,classification_report, ConfusionMatrixDisplay
print('~~~~~~~ TRAINING ~~~~~~~')
print("Accuracy = ", accuracy_score(train_lables, predicted_train))

report = classification_report(le.inverse_transform(y_train), predicted_train)
print(report)

print('Confusion Matrix :')
string1="airplane,apple,banana,bicycle,car,cat,chair,duck,teddy bear,pizza,fire hydrant,train,elephant,knife,cup"
class_names=sorted(list(string1.split(",")))
confusion_matrix=confusion_matrix(train_lables,predicted_train,labels=class_names)

import seaborn as sns
import matplotlib.pyplot as plt
plt.subplots(figsize=(15,15))

seaborn1=sns.heatmap(confusion_matrix, xticklabels=class_names, yticklabels=class_names, annot=True, annot_kws={'size': 10},cmap='Blues')

# Save confusion matrix and report
results_path = '../working/train_confusion_vgg.png'
plt.savefig(results_path)
report_path = '../working/training_report_vgg.txt'
text_file = open(report_path, "w")
n = text_file.write(report)
text_file.close()


# In[ ]:


test_features_extractor =model.predict(x_test)
test_feature = test_features_extractor.reshape(test_features_extractor.shape[0],-1)


# In[ ]:


predicted_output = svmmodel.predict(test_feature)
predicted_output = le.inverse_transform(predicted_output)


# In[ ]:


from sklearn.metrics import accuracy_score,confusion_matrix
print('~~~~~~~ TESTING ~~~~~~~')
print("Accuracy = ", accuracy_score(test_lables, predicted_output))
from sklearn.metrics import classification_report
report = classification_report(le.inverse_transform(y_test), predicted_output)
print(report)

confusion_matrix=confusion_matrix(le.inverse_transform(y_test), predicted_output)

import seaborn as sns
import matplotlib.pyplot as plt
plt.subplots(figsize=(15,15))
string1="airplane,apple,banana,bicycle,car,cat,chair,duck,teddy bear,pizza,fire hydrant,train,elephant,knife,cup"
class_names=sorted(list(string1.split(",")))

seaborn1=sns.heatmap(confusion_matrix, xticklabels=class_names, yticklabels=class_names, annot=True, annot_kws={'size': 10},cmap='Blues')

# Save confusion matrix and report
results_path = '../working/test_confusion_vgg.png'
plt.savefig(results_path)
report_path = '../working/testing_report_vgg.txt'
text_file = open(report_path, "w")
n = text_file.write(report)
text_file.close()


# In[ ]:


val_images = []
val_lables = []
for directory_path in glob.glob("../input/sbir151/data/test/*"):
    lables = directory_path.split("/")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        val_images.append(img)
        val_lables.append(lables)
val_images = np.array(val_images)
val_lables = np.array(val_lables)


# In[ ]:


val_features_extractor = model.predict(val_images)
val_feature = val_features_extractor.reshape(val_features_extractor.shape[0],-1)


# In[ ]:



predicted_output_val = svmmodel.predict(val_feature)
predicted_output_val = le.inverse_transform(predicted_output_val)


# In[ ]:


from sklearn.metrics import accuracy_score,confusion_matrix
print('~~~~~~ Validation ~~~~~~~')
print("Accuracy = ", accuracy_score(val_lables, predicted_output_val))
from sklearn.metrics import classification_report
report = classification_report(val_lables, predicted_output_val)
print(report)
print('Confusion Matrix :')
confusion_matrix=confusion_matrix(val_lables,predicted_output_val)
print(confusion_matrix)

import seaborn as sns
import matplotlib.pyplot as plt
plt.subplots(figsize=(15,15))
string1="airplane,apple,banana,bicycle,car,cat,chair,duck,teddy bear,pizza,fire hydrant,train,elephant,knife,cup"
class_names=sorted(list(string1.split(",")))

seaborn1=sns.heatmap(confusion_matrix, xticklabels=class_names, yticklabels=class_names, annot=True, annot_kws={'size': 10},cmap='Blues')

# Save confusion matrix and report
results_path = '../working/valid_confusion_vgg.png'
plt.savefig(results_path)
report_path = '../working/validation_vgg.txt'
text_file = open(report_path, "w")
n = text_file.write(report)
text_file.close()


# In[ ]:


from matplotlib import pyplot as plt


# In[ ]:


n=np.random.randint(0, x_test.shape[0])
img = x_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0)
input_img_feature = model.predict(input_img)
input_img_features = input_img_feature.reshape(input_img_feature.shape[0],-1)
prediction_SVM = svmmodel.predict(input_img_features)[0]
prediction_SVM = le.inverse_transform([prediction_SVM])
print('Predicted Label:')
print(prediction_SVM)
print('Actual Label:')
print(test_lables[n])


# In[ ]:


import tensorflow as tf
import cv2
import glob
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import numpy as np
import os,pickle, matplotlib.pyplot as plt
model = VGG16(weights='imagenet',include_top=False,input_shape=(100,100,3))

with open('../input/model11/svm-vgg-16.h5', 'rb') as file:
    cnn = pickle.load(file)
string1="airplane,apple,banana,bicycle,car,cat,chair,duck,teddy bear,pizza,fire hydrant,train,elephant,knife,cup"
class_names=sorted(list(string1.split(",")))
img = cv2.imread('../input/sbir151/data/val/fire_hydrant/1206.png')
plt.imshow(img)
input_img = np.expand_dims(img, axis=0)
input_img_feature = model.predict(input_img)
input_img_features = input_img_feature.reshape(input_img_feature.shape[0],-1)
prediction_SVM = cnn.predict(input_img_features)[0]
prediction=class_names[prediction_SVM]
print('Predicted Label:')
print(prediction)
print('Actual Label:')
print('Fire_Hydrant')


# In[ ]:


n=np.random.randint(0, x_test.shape[0])
img = x_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0)
input_img_feature = model.predict(input_img)
input_img_features = input_img_feature.reshape(input_img_feature.shape[0],-1)
prediction_SVM = svmmodel.predict(input_img_features)[0]
prediction_SVM = le.inverse_transform([prediction_SVM])
print('Predicted Label:')
print(prediction_SVM)
print('Actual Label:')
print(test_lables[n])


# In[ ]:





# In[ ]:




