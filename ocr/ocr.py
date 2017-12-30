
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


def ocr_by_tesseract(img, boundingBox):
	(x, y, w, h) = boundingBox;
	if False:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		textImg = gray[y:y+h, x:x+w]
		textImg = cv2.medianBlur(textImg, 3)
		textImg = basics.threshold_img(textImg, 'adaptive')
	else:
		textImg = img[y:y+h, x:x+w]
	cv2.imwrite('bib.png', textImg)
	text = pytesseract.image_to_string(Image.open('bib.png'))
	return text

def ocr_by_BBP(img, boundingBox):
	(x, y, w, h) = boundingBox;
	textImg = img[y:y+h, x:x+w]
	features = bbpDesc.describe(textImg).reshape(1, -1)
	prediction = bbpDigitModel.predict(features)[0]
	return prediction

def ocr(img, method='bbp', visualize=False):

	h, w = img.shape[:2]

	if len(img.shape)==3 and img.shape[2]>1:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		thresh = basics.threshold_img(gray, 'adaptive')
	else:
		thresh = img
	contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[1]

	cnt_img = img.copy()
	text = ''
	for cnt in contours:
		(x, y, w, h) = cv2.boundingRect(cnt)
		rect = cv2.minAreaRect(cnt)

		ar = float(w)/float(h)
		
		box = cv2.boxPoints(rect)
		box = np.int0(box)

		cv2.rectangle(cnt_img, (x, y), (x+w-1, y+h-1), (0, 0, 255), CONTOUR_WIDTH)
		if (method == 'bbp'):
			text = text + ocr_by_BBP(img, (x, y, w, h))
		elif (method == 'tesseract'):
			text = text + ocr_by_tesseract(img, (x, y, w, h))

	if (visualize == True):
		basics.showImageVertical([img, thresh, cnt_img], 'debug_ocr', 0, 1000, 0, 0, 0)

	return text