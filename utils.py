import cv2
import glob
import numpy as np
import os
import pandas as pd


label_map = {'SANO':0,'MODERADA':1,'SEVERA':2}

def img_dir_to_nparray(dir):
	X = []
	files = glob.glob(dir) 
	for file in files:
		image = cv2.imread(file, cv2.IMREAD_COLOR)
		image = cv2.resize(image,(90,90))
		X.append(image)

	return np.array(X)


def load_data_dir(csv_file,image_dir):
	"""
	Return np array X od image data  and labels Y using the csv file and the 
	image dir
	"""
	data = pd.read_csv(csv_file)
	images_labels = data['estado'].tolist()
	images_names = data['nombre'].tolist()
	# print (images_labels)
	X = []
	Y = []
	for i in range(len(images_names)):
		image_file = os.path.sep.join([image_dir,str(images_names[i]) + '.png'])
		image = cv2.imread(image_file,cv2.IMREAD_COLOR)
		image = cv2.resize(image,(90,90))
		X.append(image)
		label = label_map[images_labels[i]]
		Y.append(label)

	return np.array(X),np.array(Y)