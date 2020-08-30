from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from mymodule.simpledatasetloader import SimpleDatasetLoader
from mymodule.imagetoarraypreprocessor import  ImageToArrayPreprocessor
from mymodule.simpleprocessor import SimpleProcessor
from mymodule.lenet import LeNet

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint

from imutils import paths
import numpy as np
import pandas as pd
import argparse
import cv2


data_path = 'data'

im_paths = list(paths.list_images(data_path))
sp = SimpleProcessor(28,28)
sdl = SimpleDatasetLoader(preprocessors =[sp])


(data,labels) = sdl.load(im_paths,verbose=500)


data = np.expand_dims(data,axis = 3)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
data = data.astype('float32') / 255.0

X_train,X_val,y_train,y_val = train_test_split(data,labels,test_size = 0.15)

data_gen_args = {
            width_shift_range = 0.3, 
            height_shift_range = 0.3,
            zoom_range = 0.5,
            fill_mode = 'nearest'
}

X_datagen = ImageDataGenerator(**data_gen_args)
X_datagen.fit(X_train)

X_datagen_val = ImageDataGenerator(**data_gen_args)
X_datagen_val.fit(X_val)

cp_1 = ModelCheckpoint('best_model_lenet_aug_loss_3.model', monitor = 'val_loss', mode = 'min',
                     save_best_only = True, verbose = 1)

cp_2 = ModelCheckpoint('best_model_lenet_aug_acc_3.model', monitor = 'val_acc', mode = 'max',
                     save_best_only = True, verbose = 1)

model = LeNet.build(28,28,1,9)

model.compile(optimizer= "adam",loss = "categorical_crossentropy",metrics=["accuracy"])
H = model.fit_generator(X_datagen.flow(X_train, y_train, batch_size=32),
                        validation_data = X_datagen_val.flow(X_val, y_val, batch_size=32),
                        callbacks = [cp_1, cp_2], steps_per_epoch = len(X_train) / 32,
                        epochs = 80)