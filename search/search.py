from __future__ import print_function

import numpy as np
import cv2
import sys
import os

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path + '/../')

import image_processing.basics as basics
#template must be a canny image
#target must be a gray level image
def searchImageByMatchTemplate(template, target, searchRegion, threshold, bVisualize=False):

	(x0, y0, x1, y1) = searchRegion
	if (x0, y0, x1, y1) == (0, 0, 0, 0):
		x1 = target.shape[1]
		y1 = target.shape[0]
	target = target[y0:y1, x0:x1]

	(templateH, templateW) = template.shape[:2]
	(tH, tW) = target.shape[:2]
	
	bFound = False

	edged = cv2.Canny(target, 50, 200)
	try:
		result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF_NORMED)
		(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
	except:
		print('target= ', (tW, tH)) 
		print('Template= ', (templateW, templateH))
		return (False, 0, (0, 0, 0, 0))

	r = 1
	if maxVal > threshold:
		#print('found: maxval = ', maxVal, 'r=', r)
		bFound = True

	(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
	(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

	crop_W = int(templateW*r)
	crop_H = int(templateH*r)

	if bVisualize == True:
		cv2.rectangle(scaledTarget, (x, y), (x+w-1, y+h-1), (255, 0, 0), 5)
		basics.showResizeImg(scaledTarget, targetName, 0, 0, 0, 0, 0)

	return (bFound, maxVal, (startX, startY, crop_W, crop_H))

def searchImageByHOG(template, target, searchRegion, threshold, HOGParam, SearchStep, bVisualize=False):

	(x0, y0, x1, y1) = searchRegion
	if (x0, y0, x1, y1) == (0, 0, 0, 0):
		x1 = target.shape[1]
		y1 = target.shape[0]
	target = target[y0:y1, x0:x1]

	(templateH, templateW) = template.shape[:2]
	(tH, tW) = target.shape[:2]

	(orientations, pixels_per_cell, cells_per_block, transform_sqrt, block_norm) = HOGParam


	return (bFound, maxVal, (startX, startY, crop_W, crop_H))

