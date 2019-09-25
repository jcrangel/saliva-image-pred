# Load libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from utils import *
from models import VGG19
from sklearn.model_selection import KFold

IMAGE_DIMS = (90,90,3)
BS = 8
EPOCHS = 10

X,Y = load_data_dir('data/datamap.csv','image-data')
X = X.astype('float32')
X /= 255 
Y = tf.keras.utils.to_categorical(Y,3)

print ('X shape: ', X.shape)
print ('Y shape: ', Y.shape)


 
n_split=3

losses = []
accuracies = [] 
for train_index,test_index in KFold(n_split).split(X):
  x_train,x_test=X[train_index],X[test_index]
  y_train,y_test=Y[train_index],Y[test_index]
  
  model = VGG19.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
    depth=IMAGE_DIMS[2], classes=3,finalAct='softmax')  
  model.fit(x_train, y_train,epochs=10)
  
  res =  model.evaluate(x_test,y_test)
  loss = res[0]
  acc = res[1]
  print('Model evaluation ',res)
  losses.append(loss)
  accuracies.append(acc)

print('Mean accuracy: ' , np.mean(accuracies))
print('Mean loss: ' , np.mean(loss))





