# Python Script to Capture a video and Diaplay it until q is pressed

import cv2
import numpy as np

#Defining a Name for Camera
Capture = cv2.VideoCapture(0)

#Name of Video to be Recorded
VideoName = 'Out'
four_cc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(VideoName, four_cc, 10.0, (640,480)) 

#Input to Capture and Save Video or Not
Record = True

if __name__=='__main__':

    print 'Starting..'
    #Starting Capturing Frames and Display it
    while (Capture.isOpened()):

        print 'Reading Frame..'
        success, frame = Capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        if not success:
            print 'No Frames Available'
            break
        
        cv2.imshow("Frames", frame)
        #Save the Frames in Video
        if Record is True:
            video.write(frame)
            
        #Check Whether q is Pressed or Not??
        key  = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()
    Capture.release()

    

    
