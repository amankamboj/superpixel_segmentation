import cv2
import numpy as np

import time

st_time=time.time()
print(st_time)
img = cv2.imread("HR.jpg")
lsc = cv2.ximgproc.createSuperpixelLSC(img,200)
lsc.iterate(10)
mask_lsc = lsc.getLabelContourMask()
label_lsc = lsc.getLabels()
number_lsc = lsc.getNumberOfSuperpixels()
mask_inv_lsc = cv2.bitwise_not(mask_lsc)
img_lsc = cv2.bitwise_and(img,img,mask = mask_inv_lsc)
end_time=time.time()-st_time
print('Totatl time by slic'+str(end_time))
cv2.imwrite("img_lsc.jpg",img_lsc)
cv2.waitKey(0)
cv2.destroyAllWindows()