
from image_processing import basics
from ocr import ocr
import cv2
import os
import numpy as np
import sys 
from fileOp.imgReader import ImageReader

path = os.path.dirname(os.path.abspath(__file__))

class TestImgReader():
	filePath = path+'/testcase/imgReader'

	imgReader = ImageReader(filePath, False)

	while True: 
		(ret, frame, fname) = imgReader.read()
		if ret == False:
			break;
		print('fname = {}'.format(fname))

	videoReader = ImageReader('webcam', True)
	while True: 
		(ret, frame, fname) = videoReader.read()
		if ret == False:
			break;
		key = basics.showResizeImg(frame, 'webcam', 1)
		if key == ord('q'):
			break

def main():
	testcase = TestImgReader()


main()