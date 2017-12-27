
from __future__ import print_function



import numpy as np
import cv2    
import csv
import os 

import pytesseract
from PIL import Image

import sys
path = os.path.dirname(os.path.abspath(__file__))
path = path + '/../image_processing'
print('file path={}'.format(path))
sys.path.append(path)

import basics

CONTOUR_WIDTH = 2


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

def ocr(img, visualize=False):

	h, w = img.shape[:2]

	if False:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		thresh = basics.threshold_img(gray, 'adaptive')
	else:
		thresh = img
	contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[1]

	print('contours = {}'.format(len(contours)))
	cnt_img = img.copy()
	text = ''
	for cnt in contours:
		(x, y, w, h) = cv2.boundingRect(cnt)
		#print('boundingRect = {}'.format((x, y, w, h)))
		rect = cv2.minAreaRect(cnt)

		ar = float(w)/float(h)
		#print('ar = {}'.format(ar))
		
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		#print('minAreaRect = {}'.format(box))

		cv2.rectangle(cnt_img, (x, y), (x+w-1, y+h-1), (0, 0, 255), CONTOUR_WIDTH)
		text = text + ocr_by_tesseract(img, (x, y, w, h))
		cnt_img = img.copy()
		cv2.drawContours(cnt_img, [cnt], -1, (0, 255, 0), CONTOUR_WIDTH)	
		basics.showResizeImg(cnt_img, 'contour', 0, 1000, h*2, 0, 0)

	print('text = {}'.format(text))
	basics.showImageVertical([img, thresh, cnt_img], 'debug_ocr', 0, 1000, 0, 0, 0)

	#cv2.drawContours(img, contours, -1, (0, 255, 0), CONTOUR_WIDTH)	
	#basics.showResizeImg(img, 'contour', 0, 1000, h*2, 0, 0)

	#return mask