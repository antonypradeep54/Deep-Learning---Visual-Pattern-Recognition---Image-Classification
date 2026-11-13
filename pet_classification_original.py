# Original PET Classification using CNN — extracted from HTML course project.
# Source: Project - Pet Classification Model Using CNN.html
# Historical code preserved; not modernized, corrected, or execution-tested.
# Requires original CSV/image datasets and compatible dependencies.
# Cell boundaries follow the original HTML notebook order.

# %% [Original code cell 1]
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

# %% [Original code cell 2]
df_train_dataset = pd.read_csv('train_dog_cat.csv')
df_train_dataset.head(5)

# %% [Original code cell 3]
print(df_train_dataset.groupby(['Class']).size())

# %% [Original code cell 4]
df_test_dataset = pd.read_csv('test_dog_cat.csv')
df_test_dataset.head(5)

# %% [Original code cell 5]
print(df_test_dataset.groupby(['Class']).size())

# %% [Original code cell 6]
# Display a cat image from training dataset
file_name = df_train_dataset.File_Name[20]
img = mpimg.imread(file_name)
plt.imshow(img)

# %% [Original code cell 7]
# Display a dog image from training dataset
file_name = df_train_dataset.File_Name[8]
img = mpimg.imread(file_name)
plt.imshow(img)

# %% [Original code cell 8]
pet_class_names = ['Cat','Dog']
img_size = 60
img_depth = 3 
fc_size=32
drpout=0.4

# %% [Original code cell 9]
def batch_generator(df, batchsize=32):   
    
    while True:       
                
        #Generate random numbers to pick images from dataset
        batch_nums = np.random.randint(0,df.shape[0], batchsize)
        
        #Initialize batch images array
        batch_images = np.zeros((batchsize,img_size, img_size,img_depth))
        
        #Initiate batch label array
        batch_labels = np.zeros((batchsize, len(pet_class_names)))
        
        for i in range(batchsize):
            
            #Load image
            #pet_image =  tf.keras.preprocessing.image.load_img('data/' + df.loc[batch_nums[i]]['File_Name'],
            pet_image =  tf.keras.preprocessing.image.load_img(df.loc[batch_nums[i]]['File_Name'],
                                                                  target_size=(img_size, img_size))
            #Convert to array
            pet_image = tf.keras.preprocessing.image.img_to_array(pet_image)
            
            #Get the class
            img_class = df.loc[batch_nums[i]]['Class']
            #Convert class to number
            img_class = pet_class_names.index(img_class)
            #Convert class to one hot encoding
            img_class = tf.keras.utils.to_categorical(img_class, num_classes=len(pet_class_names))
            
            #Update batch images and class arrays
            batch_images[i] = pet_image
            batch_labels[i] = img_class        
        
        yield batch_images, batch_labels

# %% [Original code cell 10]
#Initialize model, reshape & normalize data
CNN_model = tf.keras.models.Sequential()

#Adding Layers
#adding input layer
CNN_model.add(tf.keras.layers.BatchNormalization(input_shape=(img_size,img_size,3,)))

#adding the first convolution layer with 32 filters and kernel size of (5,5)
CNN_model.add(tf.keras.layers.Conv2D(32, kernel_size=(5,5), activation='relu'))

#adding first max-pooling layer with pool size[2,2] and stride 2 
CNN_model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

#adding the second convolution layer with 64 filters and kernel size of (5,5)
CNN_model.add(tf.keras.layers.Conv2D(64, kernel_size=(5,5), activation='relu'))

#adding second max-pooling layer with pool size[2,2] and stride 2
CNN_model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

#adding layers for flattening the data
CNN_model.add(tf.keras.layers.Flatten())

#adding Dense layer whose output size is 32
CNN_model.add(tf.keras.layers.Dense(fc_size, activation='relu'))

#adding Dropout layer with dropout probability 0.4
CNN_model.add(tf.keras.layers.Dropout(drpout))

#adding layers for predicting the class by doing a softmax on the output of the dropout layers
CNN_model.add(tf.keras.layers.Dense(len(pet_class_names), activation='softmax'))

# %% [Original code cell 11]
#defining loss function and metrics(accuracy)
CNN_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
CNN_model.summary()

# %% [Original code cell 12]
batchsize = 1
train_generator = batch_generator(df_train_dataset, batchsize=batchsize) #batchsize can be changed
test_generator = batch_generator(df_test_dataset, batchsize=batchsize)

# %% [Original code cell 13]
CNN_model.fit_generator(train_generator, 
                          epochs=25,
                          steps_per_epoch=df_train_dataset.shape[0],
                          validation_data=test_generator,
                          validation_steps = df_test_dataset.shape[0])

# %% [Original code cell 14]
# Importing new train dataset with more images
df_train_dataset_kaggle = pd.read_csv('train_dog_cat_kaggle.csv')

# %% [Original code cell 15]
print(df_train_dataset_kaggle.groupby(['Class']).size())

# %% [Original code cell 16]
train_generator_kaggle = batch_generator(df_train_dataset_kaggle, batchsize=batchsize)

# %% [Original code cell 17]
CNN_model.fit_generator(train_generator_kaggle, 
                          epochs=50,
                          steps_per_epoch=df_train_dataset_kaggle.shape[0],
                          validation_data=test_generator,
                          validation_steps = df_test_dataset.shape[0])

