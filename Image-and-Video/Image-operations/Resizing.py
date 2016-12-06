# import the necessary packages
import cv2

# load the image and show it
image = cv2.imread("ME.jpg", -1)
    
cv2.imshow("original", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print image.shape

# we need to keep in mind aspect ratio so the image does
# not look skewed or distorted -- therefore, we calculate
# the ratio of the new image to the old image
r = 1000.0 / image.shape[1]
dim = (1000, int(image.shape[0] * r))
 
# perform the actual resizing of the image and show it
resized = cv2.resize(image, dim, interpolation = cv2.INTER_LANCZOS4)
cv2.imshow("resized", resized)
cv2.waitKey(0)

cv2.destroyAllWindows()
