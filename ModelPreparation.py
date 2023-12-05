import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential,Model
from keras.layers import Input
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import ELU, PReLU, LeakyReLU
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
class Build_model:
    def __init__(self,train_images,test_images,train_labels,test_labels) -> None:
        self.train_images=train_images
        self.test_images=test_images
        self.train_labels=train_labels
        self.test_labels=test_labels
        self.batch_size=64
        self.epochs=3
        self.num_classes=8 #len(np.unique(self.train_labels))


    def normalise_values(self):
        #self.train_images=self.train_images.astype('float32')
        #self.tes_images_images=self.test_images.astype('float32')
        #self.train_images=self.train_images.reshape(-1,250,250,1)
        #self.test_images=self.test_images.reshape(-1,250,250,1)
        self.train_labels=to_categorical(self.train_labels) #convert to one hot encoded labels
        self.test_labels=to_categorical(self.test_labels) #convert to one hot encoded leabels
        #split train images and labels into train and validation
        train_X,valid_X,train_label,valid_label = train_test_split(self.train_images, self.train_labels, test_size=0.2, random_state=13)
        train_X=train_X/ 255. #normalise train images
        valid_X=valid_X/255. #normalise validation images
        self.test_images=self.test_images/ 255. # normalise test images

        return train_X,valid_X,train_label,valid_label
        #self.train_images=self.train_images/255.0
        #self.test_images=self.test_images/255.0
        #print(self.train_images.shape,self.test_images.shape,self.train_labels.shape,self.test_labels.shape)

    def build_model(self):
        
        detection_model=Sequential()
        detection_model.add(Conv2D(32,kernel_size=(3,3),activation='linear',input_shape=(250,250,3),padding='same'))
        detection_model.add(LeakyReLU(alpha=0.1))
        detection_model.add(MaxPooling2D((2,2),padding='same'))
        detection_model.add(Conv2D(64,(3,3),activation='linear',padding='same'))
        detection_model.add(LeakyReLU(alpha=0.1))
        detection_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        detection_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
        detection_model.add(LeakyReLU(alpha=0.1))                  
        detection_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        detection_model.add(Flatten())
        detection_model.add(Dense(128, activation='linear'))
        detection_model.add(LeakyReLU(alpha=0.1))
        detection_model.add(Dense(self.num_classes,activation='softmax')) 
        return detection_model                

    def train_model(self):
        detection_model=self.build_model()
        detection_model.summary()   
        detection_model.compile(loss=keras.losses.categorical_crossentropy
                                ,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
        
        train_X,valid_X,train_label,valid_label=self.normalise_values()
        detection_train=detection_model.fit(train_X,train_label
                                            ,batch_size=self.batch_size,epochs=self.epochs
                                            ,verbose='auto',validation_data=(valid_X,valid_label)) 
        self.detection_train=detection_train
        test_eval=detection_model.evaluate(self.test_images,self.test_labels, verbose=0) 

       # print('Test loss: ', test_eval[0])
       # print ('Test accuracy: ', test_eval[1])
    def plot_accu(self):
        accuracy= self.detection_train.history['accuracy']
        val_acc = self.detection_train.history['val_accuracy']
        loss=self.detection_train.history['loss']
        val_loss=self.detection_train.history['val_loss']
        epochs= range(len(accuracy))
        plt.plot(epochs, accuracy, 'bo-', label='Training Accuracy')
        plt.plot(epochs,val_acc,'b',label='Validation Accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
       # plt.plot(epochs, loss,'bo',label='Training Loss')
       # plt.plot(epochs, val_loss,'b',label='Validation Loss')
       # plt.title('Training and Validation loss')
        plt.show()

         



        