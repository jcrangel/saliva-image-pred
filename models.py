
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
#The same but using the functional API
class VGG19:
    @staticmethod
    def build(width, height, depth, classes, finalAct="softmax"):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        inputShape = (height, width, depth)
        chanDim = -1


        conv_base =  tf.keras.applications.VGG19(weights = "imagenet", 
        include_top=False, input_shape = inputShape)

        # Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
        for layer in conv_base.layers[:7]:
            layer.trainable = False
        

        flatten_1 = Flatten()(conv_base.output)
        dense_1 = Dense(1024,activation="relu")(flatten_1)
        dropout_1 = Dropout(0.2)(dense_1)
        dense_2 = Dense(1024,activation="relu")(dropout_1)
        dense_3 = Dense(classes,activation=finalAct)(dense_2)
        
        model = Model(inputs = conv_base.input,outputs = dense_3 )

        LR = 1e-5
        OPT = Adam(lr=LR, beta_1=0.9, beta_2=0.999, decay=0,amsgrad=False,epsilon=1e-08)
        
        model.compile(loss="categorical_crossentropy",optimizer=OPT,metrics=["accuracy"])

        return model



