

import unittest
from image_processing.four_point_transform import four_point_transform
from ocr import ocr
import cv2
import os
import cPickle
import numpy as np
import sys

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

if __name__ == '__main__':
	unittest.main()