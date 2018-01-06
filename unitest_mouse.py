
from image_processing import basics
from ocr import ocr
import cv2
import os
import numpy as np
import sys 
from userinput.mouseOp import mouseCrop, mouseHandler

path = os.path.dirname(os.path.abspath(__file__))

class TestMouseCrop():
	test_img = path+'/testcase/searchTemplateMatch/img77.jpg'
	img = basics.resizeImg(cv2.imread(test_img), 1280, 720)
	cv2.imshow('mouse crop test', img)
	cv2.waitKey(1)
	def cropCallback(self, bDone, startPos, endPos):
		print('crop = {}, {}-{}'.format(bDone, startPos, endPos))
		clone = self.img.copy()
		x0 = min(startPos[0], endPos[0])
		x1 = max(startPos[0], endPos[0])
		y0 = min(startPos[1], endPos[1])
		y1 = max(startPos[1], endPos[1])

		startPos = (x0, y0)
		endPos = (x1, y1)
		cv2.rectangle(clone, (startPos), endPos, (0, 255, 0), 2)

		cv2.imshow('mouse crop test', clone)
		cv2.waitKey(1)

	def testLDownMoveLUp(self):
		mCrop = mouseCrop(dragCallback=self.cropCallback)
		cv2.setMouseCallback("mouse crop test", mouseHandler, mCrop)
		cv2.waitKey(0)

def main():
	testcase = TestMouseCrop()
	testcase.testLDownMoveLUp()

main()