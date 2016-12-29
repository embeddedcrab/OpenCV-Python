# Python script to perform feature detection and matching

# Import necessary libraries.
import numpy as np
import cv2
#from matplotlib import pyplot as plt

# set minimum match count for matching
MIN_MATCH_COUNT = 10

# Initiate SURF object, set Hessian Threshold to 600
surf = cv2.SURF(600)
# Load face cascades
face_cascade = cv2.CascadeClassifier('PATH_OF_FILE\haarcascade_frontalface_alt.xml')

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 100)

#Flann Matcher with default parameters.
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Declare Object for Camerad
Capture = cv2.VideoCapture(0)

# camera not opened then exit from script
if( not Capture.isOpened() ):
    exit()

# i am capturing image while script is running, you can do it for a saved image in memory using cv2.imread()
# Take train image for matching
success, TrainImage = Capture.read()
# if not successful in capturing image then get it frim system
if not success :
    TrainImage = cv2.imread('QueryImage.jpg')
# save image into memory, if required and show it
#cv2.imwrite('TrainImage.jpg', TrainImage)
cv2.imshow('Train Image Frame', TrainImage)
print 'Train Image clicked \nClick any key to proceed\n\n'
# wait for a key input
cv2.waitKey(0)

# convert train image into gray scale and find faces in it
TrainImage_gray = cv2.cvtColor(TrainImage,cv2.COLOR_RGB2GRAY)
faces = face_cascade.detectMultiScale(TrainImage_gray, scaleFactor=1.3, minNeighbors=5, minSize=(20, 20))

# Draw faces and find features, i am considering single face
for (x,y,w,h) in faces:
    # Draw rectangle on faces
    cv2.rectangle(TrainImage,(x,y),(x+w,y+h),(0,0,255),2)
    # Crop faces from frame
    Cropped = TrainImage[y:y+h, x:x+w]
    # find and draw the keypoints
    kp1, des1 = surf.detectAndCompute(Cropped,None)
    TrainKeyImage = cv2.drawKeypoints(Cropped, kp1, None, (0, 255, 0), 4)
    # Show features on cropped frame
    cv2.imshow('Surf Train Image', TrainKeyImage)

print 'Press any key to start matching..\n'
cv2.waitKey(0)


# Start main Function
if __name__=='__main__':

    # Process till Camera is opened
    while(Capture.isOpened()):
        
        #Capture Frame
        Success, Frame = Capture.read()
        # Check if Frame has been grabbed or not, if not then skip furthur steps and start form capturing frame
        if not Success:
            print 'No Frames Available..'
            continue

		# convert captured frame into gray scale and find faces in it
        Frame_gray = cv2.cvtColor(Frame.copy(),cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(Frame_gray, 1.3, 6)
        
		# detect faces and draw keypoints
        for (x,y,w,h) in faces:
            # Draw rectangle on faces
            cv2.rectangle(Frame,(x,y),(x+w,y+h),(0,0,255),2)
            # Crop faces from frame
            Cropped = Frame[y:y+h, x:x+w]   
            # find and draw the keypoints
            kp2, des2 = surf.detectAndCompute(Cropped,None)
            CurrentImage = cv2.drawKeypoints(Cropped, kp2, None, (0, 255, 0), 4)
            # Show features on cropped frame
            #cv2.imshow('Flann Based Matching', CurrentImage)
            
        matches = flann.knnMatch(des1,des2,k=2)
        #Store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches :
            if m.distance < 0.7*n.distance :
                good.append(m)

		# check if necessary matches are found or not
        if len(good) > MIN_MATCH_COUNT :
            print 'Matching..\n'
        else:
            print "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
            
        # Show frame
        cv2.imshow('Frame',Frame)
        #plt.imshow(Frame),plt.show()	# will work if pyplot is installed
        
        Key = cv2.waitKey(1) & 0xFF
        # if the `q` key is pressed, break from the loop
        if Key == ord('q'):
            break

	# release all the resources and edestroy all opened windows
    Capture.release()    
    cv2.destroyAllWindows()
	# End
