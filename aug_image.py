from albumentations import (ElasticTransform, 
	Compose,GridDistortion,OneOf,GaussNoise)

from tensorflow.keras.preprocessing.image import load_img,save_img

import numpy as np

def augmentate(img):
	aug = ElasticTransform(p=0.8, alpha=120, sigma=120 * 0.06, alpha_affine=120 * 0.03)
	return aug(image=img)['image']

# def augmentate(img):
# 	aug = GaussNoise()
# 	return aug(image=img)['image']

img = load_img('image-data/33.png')
img = np.array(img)
aug_img = augmentate(img)
save_img('aug1.png',aug_img)