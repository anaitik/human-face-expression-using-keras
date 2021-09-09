# code by nitik written on 10-09-2021  at 01:52 AM
#importing libraries
import keras
from keras.preprocessing.image import ImageDataGenerator
form keras.models import Sequencial
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalisation
from keras.layers import Conv2D, MaxPooling2D#maxpooling method helps features will go the second layer which are imp and rest features are left in all layers
import os


#defining class
num_classes=5    #we will train model in 5 classes
img_rows,img_cols= 48,48#48 by 48
batch_size=8 #ram is not powerful


#taking data from the file
train_data_dir=r'C:\Users\Nitik Verma\Pictures\Saved Pictures\face-expression-recognition-dataset\train'
validation_data_dir=r'C:\Users\Nitik Verma\Pictures\Saved Pictures\face-expression-recognition-dataset\validation'

#new variable
#max size of rpg is 255 and min is 1
#we are resizing and rescaling so that we have more data to train
#we are dividing it by max unit so that when after dividing it we can scale down into particular range and our training data size will be reduced
#simply we can generate more data using less data
#rescale ,rotaion at30 degree,changing width and simultaneously length by 40% ,shuffle for shuffling data
#we used shuffle so that training doesnt happen at a time, like model will train 3000happy images and then it will train sad images,but when the diff image is entered model will train 
#that according to sad and will forget happy and if one by one we do at last it will only tain last folder and forget the previous ones
train_datagen= ImageDataGenerator( rescale=1./255,rotation_range=30, shear_range=0.3,zoom_range=0.3,
width_shift_range=0.4, length_shift_range=0.4, horizontal_flip =True, vertical_flip=True)

#after training using train data we just have to cross check by using validation data hence ,the function neede

validation_datagen=ImageDataGenerator(rescale=1./255)

#we did  flipped zoomed etc to give it to the train generator function
#this we gonna give the model to train


train_generator=train_datagen.flow_from_directory(train_data_dir,color='greyscale',target_size=(img_rows,img_col),
batch_size=batch_size,class_mode='categorical',shuffle=True)


validation_generator=validation_datagen.flow_from_directory(validation_data_dir,color='greyscale',target_size=(img_rows,img_col),
batch_size=batch_size,class_mode='categorical',shuffle=True)

#softmax activation function and it containas a threshold value if the neuron crosses that value then only it will go for output

#building the model neural network
model=sequential()
#block 1 
#we using elu activation function

#convenational layer then  activation then  normalisation in block
model.add(conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape =(img)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-2 
#neurons will only change rest is same
#instead of 32 ,we use 64 similarly next blocks
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-3

model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-4 

model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#block 5
#Flattening is converting the data into a 1-dimensional array 
#for inputting it to the next layer. We flatten the output of the convolutional layers
# to create a single long feature vector




model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-6

model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-7

model.add(Dense(num_classes,kernel_initializer='he_normal'))
model.add(Activation('softmax'))

print(model.summary())

from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


#best accuracy is saved

checkpoint = ModelCheckpoint('Emotion_little_vgg.h5',  monitor='val_loss', mode='min',
 save_best_only=True, verbose=1)

#if  model accuracy is not changing we made a stop function



earlystop = EarlyStopping(monitor='val_loss',min_delta=0,
patience=3, verbose=1,restore_best_weights=True )
  


  #reduce learning rate if model not improving means learn very slow
  # after patience of 3 rounds reduce
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=3,verbose=1,min_delta=0.0001)

callbacks = [earlystop,checkpoint,reduce_lr]

model.compile(loss='categorical_crossentropy',optimizer = Adam(lr=0.001),
 metrics=['accuracy'])

nb_train_samples = 24176
nb_validation_samples = 3006
epochs=25

history=model.fit_generator(train_generator, steps_per_epoch=nb_train_samples//batch_size,epochs=epochs, callbacks=callbacks,
validation_data=validation_generator, validation_steps=nb_validation_samples//batch_size)







