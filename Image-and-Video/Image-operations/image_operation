#Script to perform operations on Images

#Importing Necessary Libraries
import cv2
import datetime

#Read an Image
Image = cv2.imread('IMG_20160518_142110.jpg', -1)

#Change Image into Gray-scale
GrayImage = cv2.cvtColor(Image, cv2.COLOR_RGB2GRAY)
#Blur Image
BlurImage = cv2.GaussianBlur(Image, (21, 21), 0) 

#Get Current Date and Time and Display 
time = datetime.datetime.now()
print time

#Save Image with Same or Different Format
cv2.imwrite('Image.png', Image)
cv2.imwrite('Gray_Image.png', GrayImage)
cv2.imwrite('Blur_Image.png', BlurImage)

#Show Image
cv2.imshow('Image', Image)
cv2.imshow('Gray Image', GrayImage)
cv2.imshow('Blur Image', BlurImage)

#Wait for Key input then Clear All Windows
cv2.waitKey(0)
cv2.destroyAllWindows()
