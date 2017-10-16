#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 20:18:08 2017

@author: sam
"""

from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model


import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from scipy import misc
import scipy.sparse
def getProbsAndPreds(someX):
    probs = softmax(np.dot(someX,w)+b)
    preds = np.argmax(probs,axis=1)+1
    return probs,preds
def getAccuracy(someX,someY):
    prob,prede = getProbsAndPreds(someX)
    accuracy = sum(prede == someY)/(float(len(someY)))
    return accuracy
def oneHotIt(Y):
    m = Y.shape[0]
    #Y = Y[:,0]
    OHX = scipy.sparse.csr_matrix((np.ones(m), (Y, np.array(range(m)))))
    OHX = np.array(OHX.todense()).T
    return OHX
model = VGG16(weights='imagenet', include_top=True)
#for layer in model.layers[:5]:
#    layer.trainable = False
train_data = sio.loadmat('PetsTrain.mat')
labels = train_data['label']
#model = VGG16(weights='imagenet', include_top=False)
#imgs = []    
img_path = '/Users/sam/Desktop/Deep Learning/assignments/hw2/images'
os.chdir(img_path)
training_data_name = train_data['files']
a = training_data_name
training_data_name = len(training_data_name)
features_out = np.zeros((4096,training_data_name),dtype ='float32')

#%%
# Getting the feature matrix of training dataset
for i in range(training_data_name):
    aa = list(a[i])
    b = list(aa[0])
    image_name = b[0]
    temp = misc.imresize(plt.imread(image_name),[224,224])
    temp = temp[:,:,0:3]
    x = image.img_to_array(temp)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    model_extractfeatures = Model(input=model.input, output=model.get_layer('fc2').output)
    fc2_features = model_extractfeatures.predict(x)
    f = np.transpose(fc2_features.reshape((4096,1)))
    features_out[:,i] = f

# Training the classifier
# skip the feature extraction
#features = np.genfromtxt('features_out.txt')
#
#
#features_out = features
x_train = np.transpose(features_out)
x_train = x_train/np.amax(x_train)
y_train = labels
w = np.zeros([4096,37])
b = np.zeros([37])
num_examples = x_train.shape[0]
lam = 0.1
iterations = 2000
learningRate = 1
losses = []
reg = 1e-3
batch = 256
batch_iter = 3680//batch + 1


for i in range(0,iterations):
    ##MINI-BATCH
#    BB = np.arange(3680)
#    np.random.shuffle(BB)
#    for j in range(batch_iter):
##        batch_size = 10
#        B = BB[(j*batch):((j+1)*batch)]
#        x_train_batch = x_train[B,:]
#        y_train_batch = y_train[B]
    y_train = np.squeeze(y_train)
    y_mat = oneHotIt(y_train) #Next we convert the integer class coding into a one-hot representation
    y_mat = y_mat[:,1:]
        
    scores = np.matmul(x_train,w)+b #Then we compute raw class scores given our input and current weights
    probs = (np.exp(scores).T / np.sum(np.exp(scores),axis=1)).T
    m = x_train.shape[0]
    loss = -(1 / m) * np.sum(y_mat * np.log(probs)) + (lam/2)*reg*np.sum(w*w)
    
      
    # backpropate the gradient to the parameters (W,b)
    dscores = (probs-y_mat)/num_examples
    dW = np.dot(x_train.T, dscores) + reg*w
    db = np.sum(dscores, axis=0, keepdims=True)
  
    
    w = w - (learningRate * dW)
    b = b - (learningRate * db)
#    if i == 500:
#        learningRate == 1
    print(loss)
    losses.append(loss)
plt.plot(losses)

#%%
## Getting the feature matrix of test data
img_path = '/Users/sam/Desktop/Deep Learning/assignments/hw2'
os.chdir(img_path)
test_data = sio.loadmat('PetsTest.mat')
labels_test = test_data['label']
labels_test1 = np.squeeze(labels_test)
model = VGG16(weights='imagenet', include_top=True)  
img_path = '/Users/sam/Desktop/Deep Learning/assignments/hw2/images'
os.chdir(img_path)
test_data_name = test_data['files']
a = test_data_name
test_data_name = len(test_data_name)
features_out_test = np.zeros((4096,test_data_name),dtype ='float32')

for i in range(test_data_name):
    aa = list(a[i])
    b = list(aa[0])
    image_name = b[0]
    temp = misc.imresize(plt.imread(image_name),[224,224])
    temp = temp[:,:,0:3]
    x = image.img_to_array(temp)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    model_extractfeatures = Model(input=model.input, output=model.get_layer('fc2').output)
    fc2_features = model_extractfeatures.predict(x)
    f = np.transpose(fc2_features.reshape((4096,1)))
    features_out_test[:,i] = f

# skip the feature extraction
#features_out_test = np.genfromtxt('features_out_test.txt')
features_out_test = features_out_test/np.amax(features_out_test)
print('Training Accuracy: ', getAccuracy(np.transpose(features_out_test),labels_test1))