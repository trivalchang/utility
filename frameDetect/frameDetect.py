# import the necessary packages
import cv2
import os
import sys


import sys
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path + '/../')

import image_processing.basics as basics

RECT_AREA_THRESH = 0.90

def IsRectangle(c, minW, minH, areaRatio=RECT_AREA_THRESH):
	x,y,w,h = cv2.boundingRect(c)	
	if (w < minW) or (h < minH):
		return False
	a0 = w * h
	a1 = cv2.contourArea(c)
	if (float(a1)/float(a0)) > areaRatio:
		return True

	return False

def IsBoxedIn(parent, child):
	x0,y0,w0,h0 = cv2.boundingRect(parent)
	x1,y1,w1,h1 = cv2.boundingRect(child)
	if x1 <= x0 or y1 <= y0:
		return False
	if (w1 > w0) or (h1 > h0):
		return False

	return True

def FrameDetectByOneImage(img0Origin, img0, minW, minH, frameRatio=0.85, shape='rect'):
	if len(img0.shape) == 3:
		img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
	thresh = basics.threshold_img(img0, 'OTSU', False)

	_, contour, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	if len(contour) == 0:
		return (False, (0, 0, 0, 0))
	frameCandidate = []
	for (cnt, hh) in zip(contour, hierarchy[0]):
		if IsRectangle(cnt, minW, minH, frameRatio)==False:
			continue
		return (True, (cv2.boundingRect(cnt)))
	
	return (False, (0, 0, 0, 0))

def FrameDetectByDiffImages(img0Origin, img0, img1, minW, minH, frameRatio=0.85, shape='rect'):
	if len(img0.shape) == 3:
		img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
	if len(img1.shape) == 3:
		img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	diff = cv2.absdiff(img0, img1)
	return FrameDetectByOneImage(img0Origin, diff, minW, minH, frameRatio, shape)


