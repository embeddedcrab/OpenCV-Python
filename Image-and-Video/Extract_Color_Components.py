# Import Necessary Libraries
import cv2
import numpy as np

# Read an image as colored one
Image = cv2.imread('F:/2015-01-07.jpg')

# Get properties of image
Shape = Image.shape
Size = Image.size
print Shape
print Size

# Create  a black image of same size as of 'Image'.
Black = np.zeros((669,446),  np.uint8)

# Create a white images of  same size as of 'Image' for
# Bluish, greenish and Reddish  images.
BlueImage = np.ones((669,446,3),  np.uint8)
GreenImage = np.ones((669,446,3), np.uint8)
RedImage = np.ones((669,446,3), np.uint8)

# Get components of main Image into  variables
Blue = Image[:,:,0]
Green = Image[:,:,1]
Red = Image[:,:,2]

# Replacing White image channels with main Image components.
# Place Blue component of Image into 0th channel of white image
BlueImage[:,:,0] = Blue
# Place Green component of Image into First channel of white image
RedImage[:,:,1] = Green
# Place Red component of Image into Second channel of white image
GreenImage[:,:,2] = Red

# Show New formed Images.
cv2.imshow('Blue', BlueImage)
cv2.imshow('Green', GreenImage)
cv2.imshow('Red', RedImage)

# Wait for ant key then destroy all  windows.
cv2.waitKey(0)
cv2.destroyAllWindows()
