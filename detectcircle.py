import cv2 as cv
import numpy as np

def process_img(img):
	imgs = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	malam = 0
	dst1 = img.copy()

	avg_color_per_row = np.average(img, axis=0)
	avg_color = np.average(avg_color_per_row, axis=0)
	avg_color1 = (avg_color[0]+avg_color[1]+avg_color[2])/3
	if avg_color1 < 21:
	    malam = 1
	
	if malam == 1:
		equ = cv.equalizeHist(imgs)
		clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		cl1 = clahe.apply(imgs)
		imgb = cv.GaussianBlur(cl1, (5, 5), 0)
		# print("malam")
		# dst = cv.resize(imgb, (720,480))
		# cv.putText(dst, "Malam",(50, 50), cv.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv.LINE_AA)
		# cv.putText(dst, "avg : "+str(avg_color1),(50, 100), cv.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv.LINE_AA)
		# showInMovedWindow('malam', dst, 720, 0)
	else:
		imgb = cv.equalizeHist(imgs)
		#imgb = imgs
		imgb = cv.GaussianBlur(imgs, (5, 5), 0)
		#imgb = cv.GaussianBlur(imgb, (5, 5), 0)
		#imgb = cv.equalizeHist(imgb)
		# print("siang")
		# dst = cv.resize(imgb, (720,480))
		# cv.putText(dst, "siang",(50, 50), cv.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv.LINE_AA)
		# cv.putText(dst, "avg : "+str(avg_color1),(50, 100), cv.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv.LINE_AA)
		# showInMovedWindow('siang', dst, 720, 0)

	circles = cv.HoughCircles (imgb, cv.HOUGH_GRADIENT, 1, 200,
                param1 = 80,
                param2 = 50,
                minRadius = 30,
                maxRadius = 60)
	img_circ = []
	i = 0
	if circles is not None:
		circles = np.uint16(np.around(circles))
		# cc = circles
		# for i in cc[0,:]:
		# 	cv.circle(imgb,(i[0],i[1]),i[2],(0,255,0),2)
		
		# imgb = cv.resize(imgb, (200,200)) 
		# showInMovedWindow('circle', imgb, 10, 490)
		# #dst = cv.resize(img, (720,480))
		# i = i + 1
		return circles
	else: return None    


def showInMovedWindow(winname, img, x, y):
    cv.namedWindow(winname)        # Create a named window
    cv.moveWindow(winname, x, y)   # Move it to (x,y)
    cv.imshow(winname,img)