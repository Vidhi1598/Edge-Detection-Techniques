from tkinter import *
from tkinter import messagebox

import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.data import camera
from skimage.filters import roberts, sobel, scharr, prewitt


root=Tk()
root.title("Edge detection techniques")
root.geometry("500x500+500+300")
root.resizable(False,False)
root.configure(background='blue')

img = cv2.imread('cameraman.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gaussian = cv2.GaussianBlur(gray,(3,3),0)



cannFrame=Toplevel(root)
cannFrame.title("Canny Edge Detection")
cannFrame.geometry("400x400+300+200")
cannFrame.withdraw()
def c1():
	#canny
	img_canny = cv2.Canny(img,100,200)
	cv2.imshow("Original Image", img)
	cv2.imshow("Canny", img_canny)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
btnCann=Button(root,text="Canny", width=20,font=("arial",20,'bold'),command=c1)
def c2():
	cannFrame.withdraw()
	root.deiconify()
btnCannBack=Button(cannFrame,text="Back",font=("arial",20,'bold'),command=c2)
btnCann.pack(pady=10)
btnCannBack.pack(pady=10)

sobFrame=Toplevel(root)
sobFrame.title("Sobel Edge Detection")
sobFrame.geometry("400x400+300+200")
sobFrame.withdraw()
def s1():

	#sobel
	img_sobelx = cv2.Sobel(img_gaussian,cv2.CV_8U,1,0,ksize=5)
	img_sobely = cv2.Sobel(img_gaussian,cv2.CV_8U,0,1,ksize=5)
	img_sobel = img_sobelx + img_sobely
	cv2.imshow("Original Image", img)
	cv2.imshow("Sobel X", img_sobelx)
	cv2.imshow("Sobel Y", img_sobely)
	cv2.imshow("Sobel", img_sobel)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
btnSob=Button(root,text="Sobel", width=20,font=("arial",20,'bold'),command=s1)
def s2():
	sobFrame.withdraw()
	root.deiconify()
btnSobBack=Button(sobFrame,text="Back",font=("arial",20,'bold'),command=s2)
btnSob.pack(pady=10)
btnSobBack.pack(pady=10)



preFrame=Toplevel(root)
preFrame.title("Prewitts Edge Detection")
preFrame.geometry("400x400+300+200")
preFrame.withdraw()
def p1():
	
	#prewitt
	kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
	kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
	img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
	img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
	cv2.imshow("Original Image", img)
	cv2.imshow("Prewitt X", img_prewittx)
	cv2.imshow("Prewitt Y", img_prewitty)
	cv2.imshow("Prewitt", img_prewittx + img_prewitty)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
btnPre=Button(root,text="Prewitts", width=20,font=("arial",20,'bold'),command=p1)
def p2():
	preFrame.withdraw()
	root.deiconify()
btnPreBack=Button(preFrame,text="Back",font=("arial",20,'bold'),command=p2)
btnPre.pack(pady=10)
btnPreBack.pack(pady=10)

lapFrame=Toplevel(root)
lapFrame.title("Laplacian Edge Detection")
lapFrame.geometry("400x400+300+200")
lapFrame.withdraw()
def l1():
	#laplacian
	img_laplacian = cv2.Laplacian(img_gaussian,cv2.CV_8U)
	cv2.imshow("Original Image", img)
	cv2.imshow("laplacian",img_laplacian)
	cv2.waitKey(0)
	cv2.destroyAllWindows() 
btnLap=Button(root,text="Laplacian", width=20,font=("arial",20,'bold'),command=l1)
def l2():
	LapFrame.withdraw()
	root.deiconify()
btnLapBack=Button(lapFrame,text="Back",font=("arial",20,'bold'),command=l2)
btnLap.pack(pady=10)
btnLapBack.pack(pady=10)

robFrame=Toplevel(root)
robFrame.title("Robert Edge Detection")
robFrame.geometry("400x400+300+200")
robFrame.withdraw()
def r1():
	#robert
	image = camera()
	edge_roberts = roberts(image)

	fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True,
                       figsize=(8, 4))

	ax[0].imshow(edge_roberts, cmap=plt.cm.gray)
	ax[0].set_title('Roberts Edge Detection')


	for a in ax:
    		a.axis('off')

	plt.tight_layout()
	plt.show()
btnRob=Button(root,text="Robert", width=20,font=("arial",20,'bold'),command=r1)
def r2():
	RobFrame.withdraw()
	root.deiconify()
btnRobBack=Button(robFrame,text="Back",font=("arial",20,'bold'),command=r2)
btnRob.pack(pady=10)
btnRobBack.pack(pady=10)






root.mainloop()
    

