from __future__ import print_function

import numpy as np
import cv2
import sys
import os
import collections
import time

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path + '/../')

import image_processing.basics as basics
from skimage import exposure
from skimage import feature
from scipy.spatial import distance

HOGParam = collections.namedtuple('HOGParam',  'orientations pixels_per_cell cells_per_block transform_sqrt block_norm')


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

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize[1]):
		for x in xrange(0, image.shape[1], stepSize[0]):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def searchImageByHOG(template, target, searchRegion, threshold, hogParam, SearchStep, bVisualize=False):

	(x0, y0, x1, y1) = searchRegion
	if (x0, y0, x1, y1) == (0, 0, 0, 0):
		x1 = target.shape[1]
		y1 = target.shape[0]
	target = target[y0:y1, x0:x1]

	(templateH, templateW) = template.shape[:2]
	(tH, tW) = target.shape[:2]

	print('hogParam = {}'.format(hogParam))
	(hogTemplate, hogImage) = feature.hog(template, orientations=hogParam.orientations, pixels_per_cell=hogParam.pixels_per_cell, 
					cells_per_block=hogParam.cells_per_block, transform_sqrt=hogParam.transform_sqrt, block_norm=hogParam.block_norm, visualise=True)

	if (bVisualize == True):
		hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
		hogImage = hogImage.astype("uint8")	
		basics.showResizeImg(hogImage, 'template HOG', 1, 0, 0, 0, 0)

	minDist = 999.0
	for (x, y, window) in sliding_window(target, SearchStep, windowSize=(templateW, templateH)):
		clone = target.copy()
		
		winImg = target[y:y+templateH, x:x+templateW]
		if (y+templateH > tH) or (x+templateW > tW):
			continue

		if bVisualize == True:
			(hogWindow, hogImage) = feature.hog(winImg, orientations=hogParam.orientations, pixels_per_cell=hogParam.pixels_per_cell, 
					cells_per_block=hogParam.cells_per_block, transform_sqrt=hogParam.transform_sqrt, block_norm=hogParam.block_norm, visualise=True)
		else:
			hogWindow = feature.hog(winImg, orientations=hogParam.orientations, pixels_per_cell=hogParam.pixels_per_cell, 
					cells_per_block=hogParam.cells_per_block, transform_sqrt=hogParam.transform_sqrt, block_norm=hogParam.block_norm, visualise=False)

		if (bVisualize == True):
			hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
			hogImage = hogImage.astype("uint8")	
			basics.showResizeImg(hogImage, 'window HOG', 1, 500, 0, 0, 0)
			cv2.rectangle(clone, (x, y), (x + templateW, y + templateH), (0, 255, 0), 2)
			key = basics.showResizeImg(clone, 'targe', 1, 0, 200)
		else:
			key = ord(' ')
		if (len(hogTemplate) != len(hogWindow)):
			continue
		dist = distance.euclidean(hogTemplate,hogWindow)
		if (minDist > dist):
			minDist = dist
			minLoc = [x, y, x + templateW, y + templateH]
		if (key == ord('q')):
			break

	print('minDist = {}, minLoc = {}'.format(minDist, minLoc))
	if (bVisualize == True):
		clone = target.copy()
		cv2.rectangle(clone, (minLoc[0], minLoc[1]), (minLoc[2], minLoc[3]), (0, 255, 0), 2)
		key = basics.showResizeImg(clone, 'targe', 0, 0, 200)

	return (True, minDist, (minLoc[0], minLoc[1], templateW, templateH))