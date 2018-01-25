# import the necessary packages
import cv2


class mouseCrop():
	CROP_METHOD_LDOWN_DRAG_LDOWN = 1
	cropMethod = CROP_METHOD_LDOWN_DRAG_LDOWN
	dragCallbacFunc = None
	cropStartPos = (0, 0)
	cropEndPos = (0, 0)
	bCropStarted = False
	def __init__(self, method=CROP_METHOD_LDOWN_DRAG_LDOWN, dragCallback=''):
		self.cropMethod = method
		self.dragCallbacFunc = dragCallback

	def lbuttonDown(self, x, y):
		print('lbuttonDown')
		if self.bCropStarted == False:
			self.bCropStarted = True
			self.cropStartPos = (x, y)
			self.cropEndPos = (x, y)
		else:
			self.bCropStarted = False
			self.cropEndPos = (x, y)
		if self.dragCallbacFunc != None:
			bDone = not self.bCropStarted
			self.dragCallbacFunc(bDone, self.cropStartPos, self.cropEndPos)

	def rbuttonDown(self, x, y):
		print('rbuttonDown')

	def lbuttonUp(self, x, y):
		print('lbuttonUp')

	def rbuttonUp(self, x, y):
		print('rbuttonUp')

	def move(self, x, y):
		self.cropEndPos = (x, y)
		print('move')
		if self.dragCallbacFunc != None and self.bCropStarted == True:
			self.dragCallbacFunc(False, self.cropStartPos, self.cropEndPos)

def mouseHandler(event, x, y, flags, param):
	mouseOp = param
	if event == cv2.EVENT_LBUTTONDOWN:
		mouseOp.lbuttonDown(x, y)
	elif event == cv2.EVENT_LBUTTONUP:
		mouseOp.lbuttonUp(x, y)
	elif event == cv2.EVENT_MOUSEMOVE:
		mouseOp.move(x, y)