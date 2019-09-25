#
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

from albumentations import (ElasticTransform, 
	Compose,GridDistortion,OneOf,GaussNoise)

import pandas as pd
import numpy as np
import CONFIG as config
import os
from models import VGG19


def append_ext(fn):
     return fn+".png"

#Elastic and distortions from the albumentations library
def distortion_aug(img):
    m=100
    aug = ElasticTransform(p=1, alpha=m, sigma=m*0.5 , alpha_affine=m * 0.03)
    return aug(image=img)['image']




IMAGE_DIMS = (32,32,3)
BS = 8
EPOCHS = 10
PLOT_FILE = 'training_plot.png'
train_file = os.path.sep.join([config.BASE_PATH,config.TRAIN_FILE])
valid_file = os.path.sep.join([config.BASE_PATH,config.TEST_FILE])
images_path = config.IMG_PATH

traindf=pd.read_csv(train_file,dtype=str)
validdf=pd.read_csv(valid_file,dtype=str)
traindf["nombre-imagen"]=traindf["nombre-imagen"].apply(append_ext)
validdf["nombre-imagen"]=validdf["nombre-imagen"].apply(append_ext)

datagen_train=ImageDataGenerator(
	rescale=1./255.,
    horizontal_flip=True,
    preprocessing_function = distortion_aug
	)

train_generator=datagen_train.flow_from_dataframe(
dataframe=traindf,
directory=images_path,
x_col="nombre-imagen",
y_col="estado-periodontal",
batch_size=BS,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(IMAGE_DIMS[0],IMAGE_DIMS[1]))


datagen_val=ImageDataGenerator(rescale=1./255.)
valid_generator=datagen_val.flow_from_dataframe(
dataframe=validdf,
directory=images_path,
x_col="nombre-imagen",
y_col="estado-periodontal",
batch_size=BS,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(IMAGE_DIMS[0],IMAGE_DIMS[1]))

model = VGG19.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
    depth=IMAGE_DIMS[2], classes=3,finalAct='softmax')         
model.summary()

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

print ('Training phase ')
H = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=EPOCHS
)



#Plot
plt.figure()
N = EPOCHS
plt.subplot(211)
plt.title("Training Loss and accuracy")
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
#plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper left")
plt.subplot(212)
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="upper left")
plt.savefig(PLOT_FILE)
# plt.show()
