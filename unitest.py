

import unittest
from image_processing.four_point_transform import four_point_transform
from image_processing import basics
from ocr import ocr
import cv2
import os
import cPickle
import numpy as np
import sys 
from search.search import searchImageByMatchTemplate, HOGParam, searchImageByHOG

path = os.path.dirname(os.path.abspath(__file__))

class TestOCR(unittest.TestCase):
	def test4pointTransformBBP(self):
		test_img = [('testcast_4pOCR/577.png', '577'), ('testcast_4pOCR/04162.png', '04162')]
		for (fname, realText) in test_img:
			
			img = cv2.imread(path+'/'+fname)
			org = img.copy()
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			contour = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
			text = ''
			for c in contour:
				(x, y, w, h) = cv2.boundingRect(c)
				if (w*h) < 50:
					continue
				cropImg = img[y:y+h, x:x+w]
				c0 = cv2.findContours(cropImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1][0]
				if (len(c0) == 0):
					continue
				rect = cv2.minAreaRect(c0)
				box = np.int0(cv2.boxPoints(rect))
				newImg = four_point_transform(cropImg, box)
				text = text + ocr.ocr(newImg, method='bbp')
			print('{} - BBP return {}, real is {}'.format(fname, text, realText))
		self.assertTrue(True)

class TestImageMatch(unittest.TestCase):
	def testTemplateMatch(self):
		testcase_path = 'testcase_searchTemplateMatch'
		test_img = [
					('img77.jpg', 'title_pic_app_foc.png', 0.70, (0, 0, 0, 0), (578, 168, 80, 74)),
					('img77.jpg', 'template0.png', 0.66, (0, 0, 0, 0), (324, 152, 61, 73)),
					]
		for (targetName, templateName, ratio, searchRegion, realLoc) in test_img:
			targetImg = cv2.imread(path+'/testcase_searchTemplateMatch/'+targetName)
			(targetH, targetW) = targetImg.shape[:2]
			targetImgOrg = targetImg.copy()
			targetImg = cv2.cvtColor(targetImg, cv2.COLOR_BGR2GRAY)
			templateImg = cv2.imread(path+'/testcase_searchTemplateMatch/'+templateName)
			templateImg = cv2.cvtColor(templateImg, cv2.COLOR_BGR2GRAY)
			templateImg = cv2.Canny(templateImg, 50, 200)

			scaledTarget = basics.resizeImg(targetImg, int(targetW*ratio), int(targetH*ratio))
			(bFound, val, (x, y, w, h)) = searchImageByMatchTemplate(templateImg, scaledTarget, searchRegion, 0.3)	
			print('search {} in {} - val = {}, loc = {}'.format(templateName, targetName, val, (x, y, w, h)))
			self.assertEqual(realLoc, (x, y, w, h))

	def testSlidingWindowHOGMatch(self):
		test_img = [
					('img77.jpg', 'title_pic_app_foc.png', 0.70, (0, 0, 0, 0), (578, 168, 80, 74)),
					('img77.jpg', 'template0.png', 0.66, (0, 0, 0, 0), (324, 152, 61, 73)),
					]
		for (targetName, templateName, ratio, searchRegion, realLoc) in test_img:
			targetImg = cv2.imread(path+'/testcase_searcSlidingWindowHOG/'+targetName)
			(targetH, targetW) = targetImg.shape[:2]
			targetImgOrg = targetImg.copy()
			targetImg = cv2.cvtColor(targetImg, cv2.COLOR_BGR2GRAY)
			scaledTarget = basics.resizeImg(targetImg, int(targetW*ratio), int(targetH*ratio))
			templateImg = cv2.imread(path+'/testcase_searchTemplateMatch/'+templateName)
			templateImg = cv2.cvtColor(templateImg, cv2.COLOR_BGR2GRAY)
			hogParam = HOGParam(orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), transform_sqrt=True, block_norm="L1")
			print('hogParam = {}'.format(hogParam))
			searchImageByHOG(templateImg, scaledTarget, searchRegion, 0.3, hogParam, (10, 10), bVisualize=True)	

if __name__ == '__main__':
	unittest.main()