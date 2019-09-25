#Slip the data set in training, validation and test. 
# Algo clean missing label  

import pandas as pd
from sklearn.model_selection import train_test_split
import os
import CONFIG as config
import numpy as np


test_size = 0.2

#
BASE_PATH = config.BASE_PATH
ANNOT_FILE = os.path.sep.join([BASE_PATH,config.ANNOT_FILE])

#Split
ANNOT_FILE = os.path.sep.join([BASE_PATH,config.ANNOT_FILE])
data = pd.read_csv(ANNOT_FILE)

#Create a split with %[ 1 - test  ][test] 
X_train, X_test_tmp = train_test_split(data,test_size=test_size,stratify=data['estado-periodontal'])
X_train.to_csv(os.path.sep.join([BASE_PATH,config.TRAIN_FILE]),index=False)
X_test_tmp.to_csv(os.path.sep.join([BASE_PATH,config.TEST_FILE]),index=False)

# Create a split of test_tmp to test and validation 50% and %50%
# ANNOT_PATH = os.path.sep.join([BASE_PATH,'test_temp.cvs'])
# data = pd.read_csv(ANNOT_PATH)

# X_val, X_test = train_test_split(data,test_size=0.5)
# X_val.to_csv(os.path.sep.join([BASE_PATH,config.VAL_FILE]),index=False)
# X_test.to_csv(os.path.sep.join([BASE_PATH,config.TEST_FILE]),index=False)


