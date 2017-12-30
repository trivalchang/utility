
from __future__ import print_function



import numpy as np
import cv2    
import csv
import os 

import pytesseract
from PIL import Image
from blockbinarypixelsum import BlockBinaryPixelSum

import sys
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path + '/../')

import image_processing.basics as basics
import cPickle

bbpCharModel = cPickle.loads(open(path+'/bbp_char.cpickle').read())
bbpDigitModel = cPickle.loads(open(path+'/bbp_digits.cpickle').read())

CONTOUR_WIDTH = 2

blockSizes = ((5, 5), (5, 10), (10, 5), (10, 10))
bbpDesc = BlockBinaryPixelSum(targetSize=(30, 15), blockSizes=blockSizes)	


def ocr_by_tesseract(img):
	cv2.imwrite('bib.png', img)
	text = pytesseract.image_to_string(Image.open('bib.png'))
	return text

def ocr_by_BBP(img):
	(h, w) = img.shape[:2]
	ar = float(w)/float(h)
	if ((ar > 2) or (ar < 0.3)):
		return ''

	features = bbpDesc.describe(img).reshape(1, -1)
	prediction = bbpDigitModel.predict(features)[0]
	return prediction

def ocr(img, method='bbp', visualize=False):
	text = ''
	if (method == 'bbp'):
		text = ocr_by_BBP(img)
	elif (method == 'tesseract'):
		text = ocr_by_tesseract(img)

	return text