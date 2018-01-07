# import the necessary packages
import cv2
import os
import image_processing.basics as basics

class ImageReader():
	imageList = []
	bVideo = False
	index = 0
	cap = None
	def __init__(self, path, is_video):
		self.imageList = []
		self.bVideo = is_video
		self.index = 0
		if (is_video == False):
			for f in os.listdir(path):
				if f.endswith(".png") or f.endswith(".jpg"):
					self.imageList.append(path+'/'+f)
			print('imgList = {}'.format(self.imageList))
		else:
			if path == 'webcam':
				self.cap = cv2.VideoCapture(0)
			else:
				self.cap = cv2.VideoCapture(path)

	def read(self):
		if (self.bVideo == True): 
			if (self.cap.isOpened() == False):
				return (False, None, None)
			ret, frame = self.cap.read()
			return (ret, frame, None)
		else:
			if (self.index >= len(self.imageList)):
				return (False, None, None)
			img = cv2.imread(self.imageList[self.index])
			imageName = self.imageList[self.index]
			self.index = self.index + 1
			return (True, img, imageName)

	def close(self):
		self.cap.release()