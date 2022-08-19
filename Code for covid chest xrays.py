#!/usr/bin/env python
# coding: utf-8

# In[18]:


#Model to classify a covid/non covid chest x-rays
#Using same images for test and train



import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
TRAIN_DIR='C:/Users/Shubojit/Desktop/coursera.dl/COVID'
TEST_DIR='C:/Users/Shubojit/Desktop/coursera.dl/COVID'
IMG_SIZE=50                         #since all the images are not perfect square,I make it a perfecrt square,i.e 50 x 50
LR=1e-3                             #Learning rate
CNN='COVID_VS_NONCOVID-{}-{}-MODEL'.format(LR,'2conv-basic')       #model name
def label_img(img):                 #defined a function and stored the img variable inside that
    word_label=img.split('.')[0]    #return 1 after detecting covid images
    print(word_label)
    if word_label=='COVID':return[1,0]  #function for training the data
    elif word_label=='NON-COVID':return[0,1]


# In[33]:


def create_train_data():
    training_data=[]
    for img in tqdm(os.listdir(TRAIN_DIR)):      #listing my images from train directory
                  print(img)
                  label=label_img(img)
                  path=os.path.join(TRAIN_DIR,img)    #joining the paths
                  img=cv2.imread(TRAIN_DIR)            #it will read the labels
                  img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE)) #resizing images to perfect square,
                                                                                            #which will be easy to train quickly
       
    training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('training_data.npy',training_data)
    return training_data


# In[20]:


def process_test_data():      #analogous to what we did for train directory(the above cell)
    testing_data=[]
    for img in tqdm(os.listdir(TEST_DIR)):
        path=os.path.join(TEST_DIR,img)
        img_num=img.split('.')[0]
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img),np.array(label)])
        
    shuffle(testing_data)
    np.save('test_data.npy',testing_data)
    return testing_data


# In[34]:


train_data = create_train_data()   #calling create_train_data() to load the data,basically loading my data


# In[35]:


#here is my network

import tflearn
from tflearn.layers.conv import conv_2D,max_pool_2d
from tflearn.layers.core import input_data,dropout,full_connected
from tflearn.estimator import regression

convnet=input_data(shapes=[none,IMG_SIZE,IMG_SIZE,1],names='input')
convnet=conv_2d(convnet,32,5,activation='relu')
convnet=max_pool_2d(convnet,5)

convnet=conv_2d(convnet,64,5,activation='relu')
convnet=max_pool_2d(convnet,5)
convnet=fully_connected(convnet,1024,activation='relu')
convnet=dropout(convnet,0.8)

convnet=fully_connected(convnet,2,activation='softmax')
convnet=fully_connected(convnet,tensorboard_dir='log')
convnet=regression(convnet,optimizer='adam',learning_rate=LR,loss='categorical_crossentropy',name='target')


# In[22]:


model=tflearn.DNN(covnet,tensorboard_dir='log')

if os.path.exists('{}.meta'.format(CNN)):
    model.load(CNN)
    print('Model')


# In[ ]:


train=train_data[:-500]
test=test_data[-500:]

X=np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
y=[i[1] for i in train]


# In[ ]:


test_x=np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y=[i[1] for i in train]


# In[23]:


model.fit({'input':X},{'target':y},n_epoch=5,validation_set=({'input':test_x}),snapshot_step=500,show_metric=True,run_id=CNN)


# In[32]:


import matplot.pyplot as plt    
test_data=process_test_data()  #calling process_test_data() by the variable test_data
fig=plt.figure()
for num,data in enumerate(test_data[:20]) #basically,it will return all the data values one by one,I am testing 20 images at first 
            img_num=data[1]   #for covid image
            img_data=data[0]  #for non covid image
            y=fig.add_subplot(3,4,num+1)
            original=img_data
            data=img_data.reshape(IMG_SIZE,IMG_SIZE,1)   #reshaping all images in perfect square 
            model_out=model.predict([data])[0]            #classifying image one by one
            if np.argmax(model_out)==1:str_label='Covid'  #conditions for covid image
            else: str_label='Non-covid'
            y.imshow(original,Cmap='gray)             #making the images grey,though most of them are grey but still
            plt.tittle(str_label)
            y.axes.get_xaxis().set_visible(False)
            y.axes.get_yaxis().set_visible(False)
                     
plt.show()

