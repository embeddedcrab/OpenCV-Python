import cv2
import numpy as np
import datetime

ShowVideo = True

Capture = cv2.VideoCapture(0)
VideoName = 'DetectionOutput.avi'
Video = cv2.VideoWriter(VideoName, 0, 20.0, (640,480))

Kernel = np.ones((5,5), np.uint8)
#green = np.uint8([[[0, 255, 0]]])
#hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
#print hsv_green

while(Capture.isOpened()):
    success, Image = Capture.read()

    if success != True:
        print 'No Frames'
        break

    hsv_Image = cv2.cvtColor(Image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([30, 100, 100])
    upper_green = np.array([90, 255, 255])

    #thresholding Images for color green
    mask = cv2.inRange(hsv_Image, lower_green, upper_green)

    #Bitwise AND masking to find Red color in Image
    Anding = cv2.bitwise_and(Image, Image, mask = mask)
    
    # Erode and Dilate according to use
    Dilate = cv2.dilate(Anding, Kernel, iterations = 2)
    Erode = cv2.erode(Dilate, Kernel, iterations = 2)

    cv2.imshow('VideoFrames',Image)
    #cv2.imshow('mask',mask)
    #cv2.imshow('Anding', Anding)
    cv2.imshow('Result', Erode)

    if ShowVideo == True:
        Video.write(Erode)
    
    key = (cv2.waitKey(1) & 0xFF)
    if key == ord("q"):
        print 'Switching OFF..!!'
        break

Capture.release()
Video.release()
cv2.destroyAllWindows()
