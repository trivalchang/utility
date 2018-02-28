
from __future__ import print_function

import numpy as np
import cv2
import os 
import sys
from sklearn.svm import SVC
import pickle

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path + '/../')

from feature.HOG import HOG
from fileOp.h5_dataset import h5_dump_dataset

class Classifier():
	model = None
	classifier_type = None
	def __init__(self, path, type='SVC'):
		self.model = pickle.loads(open(path, "rb").read())
		self.classifier_type = type

	def predict(self, feature):
		return self.model.predict([feature])