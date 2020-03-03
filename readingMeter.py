#Nhóm 5
#Ngô Minh Hoàng - 16020064
#Nguyễn Văn Hoàng - 16020231
#Nguyễn Mạnh Cường - 16020206
#Lê Hoàng - 16020229
import numpy as np
import scipy
import cv2
import tensorflow as tf
import keras
import torch
import torchvision
import sklearn
import skimage

from sklearn.externals import joblib

from classifier import Classifier
from crop import Crop


class Reader:
	
	detector = None
	
	def __init__(self, data_folder):
		self.name = "Reader"
		self.data_folder = data_folder

	# Prepare your models
	def prepare(self):
		model = sklearn.externals.joblib.load("SVM.pkl")
		self.detector = Classifier(model)

	# Implement the reading process here
	def process(self, img):
		return self.detector.run(img)

	# Prepare your models
	def prepare_crop(self):
		model = sklearn.externals.joblib.load("SVM.pkl")
		self.detector = Classifier(model)

	# Implement the reading process here
	def crop_and_process(self, img):
		crop = Crop()
		final_img = crop.run(img)
		#plt.imshow(final_img)
		final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
		#plt.imshow(final_img)
		return self.detector.run(final_img)


def check_import():
	print("Python 3.6.7")
	print("Numpy = ", np.__version__)
	print("Scipy = ", scipy.__version__)
	print("Opencv = ", cv2.__version__)
	print("Tensorflow = ", tf.__version__)
	print("Keras = ", keras.__version__)
	print("Scikit-learn = ", sklearn.__version__)
	print("Scikit-image = ", skimage.__version__)

if __name__=="__main__":
	check_import()

"""
Using TensorFlow backend.
Python 3.6.7
Numpy =	 1.14.5
Scipy =	 1.2.1
Opencv =  4.1.1
Tensorflow =  1.14.0
Keras =	 2.3.0
pytorch =  1.0.1.post2
Torch vision =	0.2.2
Scikit-learn =	0.21.3
Scikit-image =	0.14.2
"""
