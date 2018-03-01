

import unittest
from image_processing.four_point_transform import four_point_transform
from image_processing import basics
from ocr import ocr
import cv2
import os
import cPickle
import numpy as np
import sys 
from search.search import searchImageByMatchTemplate, HOGParam, searchImageByHOG, searchImageByHOGFeature
from feature.HOG import HOG

path = os.path.dirname(os.path.abspath(__file__))

class TestOCR(unittest.TestCase):
	def test4pointTransformBBP(self):

		return
		test_img = [('testcase/4pOCR/577.png', '577'), ('testcase/4pOCR/04162.png', '04162')]
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

		return

		testcase_path = 'testcase/searchTemplateMatch'
		test_img = [
					('img77.jpg', 'title_pic_app_foc.png', 0.70, (0, 0, 0, 0), (578, 168, 80, 74)),
					('img77.jpg', 'template0.png', 0.66, (0, 0, 0, 0), (324, 152, 61, 73)),
					]
		for (targetName, templateName, ratio, searchRegion, realLoc) in test_img:
			targetImg = cv2.imread(path+'/testcase/searchSlidingWindowHOG/'+targetName)
			(targetH, targetW) = targetImg.shape[:2]
			targetImgOrg = targetImg.copy()
			targetImg = cv2.cvtColor(targetImg, cv2.COLOR_BGR2GRAY)
			templateImg = cv2.imread(path+'/testcase/searchSlidingWindowHOG/'+templateName)
			templateImg = cv2.cvtColor(templateImg, cv2.COLOR_BGR2GRAY)
			templateImg = cv2.Canny(templateImg, 50, 200)

			scaledTarget = basics.resizeImg(targetImg, int(targetW*ratio), int(targetH*ratio))
			(bFound, val, (x, y, w, h)) = searchImageByMatchTemplate(templateImg, scaledTarget, searchRegion, 0.3)	
			print('search {} in {} - val = {}, loc = {}'.format(templateName, targetName, val, (x, y, w, h)))
			self.assertEqual(realLoc, (x, y, w, h))

	def testSlidingWindowHOGMatch(self):

		return

		test_img = [
					('img77.jpg', 'title_pic_foc.png', 0.7, 1, (0, 0, 0, 0), (578, 168, 80, 74)),
					#('img77.jpg', 'title_pic_audio_foc.png', 0.7, 1, (0, 0, 0, 0), (578, 168, 80, 74)),
					#('img77.jpg', 'title_pic_app_foc.png', 0.7, 1, (0, 0, 0, 0), (578, 168, 80, 74)),
					#('img77.jpg', 'title_pic_sx_foc.png', 0.7, 1, (0, 0, 0, 0), (578, 168, 80, 74)),
					#('img77.jpg', 'title_pic_setup_foc.png', 0.7, 1, (0, 0, 0, 0), (578, 168, 80, 74)),
					#('img77.jpg', 'template0.png', 0.66, 1, (0, 0, 0, 0), (324, 152, 61, 73)),
					#('img77.jpg', 'template1.png', 0.66, 1, (0, 0, 0, 0), (324, 152, 61, 73)),
					#('img77.jpg', 'template2.png', 0.66, 1, (0, 0, 0, 0), (324, 152, 61, 73)),
					#('img77.jpg', 'template3.png', 0.66, 1, (0, 0, 0, 0), (324, 152, 61, 73)),
					#('img77.jpg', 'template4.png', 0.66, 1, (0, 0, 0, 0), (324, 152, 61, 73)),
					#('img77.jpg', 'template5.png', 0.66, 1, (0, 0, 0, 0), (324, 152, 61, 73)),
					#('img77.jpg', 'template6.png', 0.66, 1, (0, 0, 0, 0), (324, 152, 61, 73)),
					]
		for (targetName, templateName, targetRatio, templateRatio, searchRegion, realLoc) in test_img:
			targetImg = cv2.imread(path+'/testcase/searchSlidingWindowHOG/'+targetName)
			(targetH, targetW) = targetImg.shape[:2]
			targetImg = basics.resizeImg(targetImg, int(targetW*targetRatio), int(targetH*targetRatio))
			targetImgColor = targetImg.copy()
			targetImg = cv2.cvtColor(targetImg, cv2.COLOR_BGR2GRAY)
			targetImg = basics.blur_img(targetImg, 'gussian', (5,5))
			templateImg = cv2.imread(path+'/testcase/searchSlidingWindowHOG/'+templateName)
			templateImg = cv2.cvtColor(templateImg, cv2.COLOR_BGR2GRAY)
			templateImg = basics.resizeImg(templateImg, int(templateImg.shape[1]*templateRatio), int(templateImg.shape[0]*templateRatio))
			hogParam = HOGParam(orientations=9, 
								pixels_per_cell=(8,8), 
								cells_per_block=(2,2), 
								transform_sqrt=True, 
								block_norm="L2")
			templateAR = float(templateImg.shape[1])/float(templateImg.shape[0])
			stepSize = (10, int(10/templateAR))
			(bFound, val, (x, y, w, h)) = searchImageByHOG(templateImg, 
														targetImg, 
														searchRegion, 
														0.3, 
														hogParam, 
														stepSize, 
														bVisualize=False)
			print('testSlidingWindowHOGMatch search {} in {} - val = {}, loc = {}'.format(templateName, targetName, val, (x, y, w, h)))
			cv2.rectangle(targetImgColor, (x, y), (x+w, y+h), (0, 255, 0), 2)
			key = basics.showResizeImg(targetImgColor, 'targe', 0, 0, 200)

	def testSlidingWindowHOGMatchByFeature(self):

		test_img = [
					('img77.jpg', 'title_pic_foc.png', 0.7, 1, (0, 0, 0, 0), (578, 168, 80, 74)),
					#('img77.jpg', 'title_pic_audio_foc.png', 0.7, 1, (0, 0, 0, 0), (578, 168, 80, 74)),
					#('img77.jpg', 'title_pic_app_foc.png', 0.7, 1, (0, 0, 0, 0), (578, 168, 80, 74)),
					#('img77.jpg', 'title_pic_sx_foc.png', 0.7, 1, (0, 0, 0, 0), (578, 168, 80, 74)),
					#('img77.jpg', 'title_pic_setup_foc.png', 0.7, 1, (0, 0, 0, 0), (578, 168, 80, 74)),
					#('img77.jpg', 'template0.png', 0.66, 1, (0, 0, 0, 0), (324, 152, 61, 73)),
					#('img77.jpg', 'template1.png', 0.66, 1, (0, 0, 0, 0), (324, 152, 61, 73)),
					#('img77.jpg', 'template2.png', 0.66, 1, (0, 0, 0, 0), (324, 152, 61, 73)),
					#('img77.jpg', 'template3.png', 0.66, 1, (0, 0, 0, 0), (324, 152, 61, 73)),
					#('img77.jpg', 'template4.png', 0.66, 1, (0, 0, 0, 0), (324, 152, 61, 73)),
					#('img77.jpg', 'template5.png', 0.66, 1, (0, 0, 0, 0), (324, 152, 61, 73)),
					#('img77.jpg', 'template6.png', 0.66, 1, (0, 0, 0, 0), (324, 152, 61, 73)),
					]
		for (targetName, templateName, targetRatio, templateRatio, searchRegion, realLoc) in test_img:
			targetImg = cv2.imread(path+'/testcase/searchSlidingWindowHOG/'+targetName)
			(targetH, targetW) = targetImg.shape[:2]
			targetImg = basics.resizeImg(targetImg, int(targetW*targetRatio), int(targetH*targetRatio))
			targetImgColor = targetImg.copy()
			targetImg = cv2.cvtColor(targetImg, cv2.COLOR_BGR2GRAY)
			targetImg = basics.blur_img(targetImg, 'gussian', (5,5))
			templateImg = cv2.imread(path+'/testcase/searchSlidingWindowHOG/'+templateName)
			templateImg = cv2.cvtColor(templateImg, cv2.COLOR_BGR2GRAY)
			templateImg = basics.resizeImg(templateImg, int(templateImg.shape[1]*templateRatio), int(templateImg.shape[0]*templateRatio))

			templateHOG = HOG(	_orientations=9, 
								_pixels_per_cell=(8,8), 
								_cells_per_block=(2,2), 
								_transform_sqrt=True, 
								_block_norm="L2")

			(templateFeature, _) = templateHOG.describe(templateImg)

			hogParam = HOGParam(orientations=9, 
								pixels_per_cell=(8,8), 
								cells_per_block=(2,2), 
								transform_sqrt=True, 
								block_norm="L2")
			templateAR = float(templateImg.shape[1])/float(templateImg.shape[0])
			stepSize = (10, int(10/templateAR))
			e1 = cv2.getTickCount()
			(bFound, val, (x, y, w, h)) = searchImageByHOGFeature(	templateFeature, 
																	templateImg.shape,
																	targetImg, 
																	searchRegion, 
																	0.3, 
																	hogParam, 
																	stepSize,
																	bMP=False, 
																	bVisualize=False)
			e2 = cv2.getTickCount()
			time = (e2 - e1)/ cv2.getTickFrequency()
			print('testSlidingWindowHOGMatchByFeature search {} in {} - val = {}, loc = {}, takes {}'.format(templateName, targetName, val, (x, y, w, h), time))			

			e1 = cv2.getTickCount()
			(bFound, val, (x, y, w, h)) = searchImageByHOGFeature(	templateFeature, 
																	templateImg.shape,
																	targetImg, 
																	searchRegion, 
																	0.3, 
																	hogParam, 
																	stepSize,
																	bMP=True, 
																	bVisualize=False)
			e2 = cv2.getTickCount()
			time = (e2 - e1)/ cv2.getTickFrequency()
			print('testSlidingWindowHOGMatchByFeature(MP) search {} in {} - val = {}, loc = {}, takes {}'.format(templateName, targetName, val, (x, y, w, h), time))			

			#cv2.rectangle(targetImgColor, (x, y), (x+w, y+h), (0, 255, 0), 2)
			#key = basics.showResizeImg(targetImgColor, 'targe', 0, 0, 200)



if __name__ == '__main__':
	unittest.main()