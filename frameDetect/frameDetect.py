# import the necessary packages
import cv2

RECT_AREA_THRESH = 0.98

def IsRectangle(c, minW, minH):
	x,y,w,h = cv2.boundingRect(c)	
	if (w < minW) or (h < minH):
		return False
	a0 = w * h
	a1 = cv2.contourArea(c)
	if (float(a1)/float(a0)) > RECT_AREA_THRESH:
		return False

	return True



def frameDetect(img0, img1, minW, minH, frameRatio=0.85, shape='rect'):

	img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
	img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	delta = cv2.absdiff(img0, img1)
	T, thresh = cv2.threshold(delta, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	_, contour, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	for (cnt, hh) in zip(contour, hierarchy[0]):

		if IsRectangle(cnt, minW, minH)==False:
			continue

		# check if there is a child. If no child, skip
		if (hh[2] == -1):
			continue

		child_cnt = contour[hh[2]]
		if IsRectangle(cnt, int(minW*frameRatio), int(minH*frameRatio))==False:
			continue

		outerArea = cv2.contourArea(cnt)
		innerArea = cv2.contourArea(child_cnt)
		if (innerArea >= frameRatio * outerArea):
			return (True, (cv2.boundingRect(cnt)))
	return (False, (0, 0, 0, 0))

