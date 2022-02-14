import cv2
import numpy as np
import time

st_time=time.time()
print(st_time)
img = cv2.imread("HR.jpg")
#Initialize the slic item, the average size of super pixels is 20 (default is 10), and the smoothing factor is 20
slic = cv2.ximgproc.createSuperpixelSLIC(img,region_size=250,ruler = 40.0) 
slic.iterate(10)     #Number of iterations, the greater the better
mask_slic = slic.getLabelContourMask() #Get Mask, Super pixel edge Mask==1
label_slic = slic.getLabels()        #Get superpixel tags
number_slic = slic.getNumberOfSuperpixels()  #Get the number of super pixels
mask_inv_slic = cv2.bitwise_not(mask_slic)  
img_slic = cv2.bitwise_and(img,img,mask =  mask_inv_slic) #Draw the superpixel boundary on the original image
end_time=time.time()-st_time
print('Totatl time by slic'+str(end_time))
cv2.imwrite("img_slic.jpg",img_slic)
cv2.waitKey(0)
cv2.destroyAllWindows()