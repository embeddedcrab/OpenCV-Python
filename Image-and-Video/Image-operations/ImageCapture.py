# Python Script to Capture Frames and Display it until q is pressed

import cv2
import numpy
import datetime
import time

#Defining a Name for Camera
Capture = cv2.VideoCapture(0)

#Name of Images Starting from 1
ImageNumber = 1


if __name__=='__main__':

    print 'Starting..'
    
    #Starting Capturing Frames and Display it
    while (True):

        success, frame = Capture.read()
        if not success:
            print 'No Frames Available'
            break
        
        cv2.imshow("Frames", frame)

        timestamp  = datetime.datetime.now()
        print timestamp
        #   CurrentTime = timestamp.strftime("%A %d %B %Y %I:%M:%S:%p")
        #Print Current Time on Frame
        #   cv2.putText(frame, CurrentTime, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        #           0.35, (0, 0, 255), 1)

        key  = cv2.waitKey(1) & 0xFF
        if key == ord("q"):     #Quit when key 'q' is pressed
            print 'Switching OFF..'
            break
        elif key == ord("s"):   #Save respective frame on which key 's' was pressed
            cv2.imwrite( (str(ImageNumber) + '.png'), frame)
            ImageNumber = ImageNumber + 1
            print 'Frame Saved !'
           

    # Release the Capture Buffer
    cv2.destroyAllWindows()
    Capture.release()

