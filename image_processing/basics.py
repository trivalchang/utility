
from __future__ import print_function

import numpy as np
import cv2    
import os 


def resizeImg(img, width=1280, height=720):
	(h,w) = img.shape[:2]
	print('img size is {}x{} to {}x{}'.format(w,h,width,height))
	r = min([float(width)/w, float(height)/h])
	(w, h) = (int(r * w), int(r * h))
	print('resizeImg to {}x{}'.format(w,h))
	img = cv2.resize(img, (w, h))
	return img

def showResizeImg(img, name, waitMS, x=0, y=0, width=1280, height=720):
	(h,w) = img.shape[:2]
	if (width != 0) and (height !=0):
		r = min([float(width)/w, float(height)/h])
		(w, h) = (int(r * w), int(r * h))
		img = cv2.resize(img, (w, h))
	cv2.imshow(name, img)
	cv2.moveWindow(name, x, y)
	key = cv2.waitKey(waitMS)

	return key & 0xFF

def showImageVertical(imgList, imgName, waitMS, x=0, y=0, width=720, height=480):
	
	maxW = 0
	maxH = 0
	for img in imgList:
		h, w = img.shape[:2]
		maxW = max(maxW, w)
		maxH += h
	output = np.zeros((maxH,maxW,3), dtype=np.uint8)

	nextY = 0
	for img in imgList:
		h, w = img.shape[:2]
		if len(img.shape) == 2:
			img = np.dstack([img]*3)
		output[nextY:nextY+h, 0:w] = img
		nextY += h

	showResizeImg(output, imgName, waitMS, x, y, width, height)	

def blur_img(img, method):
	if (method == 'bilateral'):
		diameter = 9
		sigmaColor = 21
		sigmaSpace = 7
		blur = cv2.bilateralFilter(img, diameter, sigmaColor, sigmaSpace)
	else:
		blur = img.copy()
	return blur

def threshold_img(img, method):
	if (method == 'adaptive'):
		thresholded = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
	elif (method == 'OTSU'):
		T, thresholded = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
	else:
		if (method != None):
			try:
				T = int(method)
			except:
				T = 128
		T, thresholded = cv2.threshold(img, T, 255, cv2.THRESH_BINARY_INV)
	return thresholded

def morphological_process(img, method, kernelSize):
	if (method == 'closing'):
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
		closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
		return closing
	elif (method == 'blackhat'):
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
		blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
		return blackhat
	elif (method == 'erode'):
		eroded = cv2.erode(img, None, iterations=kernelSize)
		return eroded
	elif (method == 'dilate'):
		dilated = cv2.dilate(img, None, iterations=kernelSize)
		return dilated
	return img.copy()
