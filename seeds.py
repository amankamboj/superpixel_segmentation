import cv2
import numpy as np
import time

st_time=time.time()
print(st_time)
img = cv2.imread("HR.jpg")
#Initialize the seeds item, pay attention to the order of the length and width of the picture
seeds = cv2.ximgproc.createSuperpixelSEEDS(img.shape[1],img.shape[0],img.shape[2],250,15,3,5,True)
seeds.iterate(img,10)  #The input image size must be the same as the initial shape, the number of iterations is 10
mask_seeds = seeds.getLabelContourMask()
label_seeds = seeds.getLabels()
number_seeds = seeds.getNumberOfSuperpixels()
mask_inv_seeds = cv2.bitwise_not(mask_seeds)
img_seeds = cv2.bitwise_and(img,img,mask =  mask_inv_seeds)
end_time=time.time()-st_time
print('Totatl time by slic'+str(end_time))
cv2.imwrite("img_seeds.jpg",img_seeds)
cv2.waitKey(0)
cv2.destroyAllWindows()