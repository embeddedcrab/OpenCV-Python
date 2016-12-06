# Python Script to Capture a video and Diaplay it until q is pressed
import cv2
import numpy

#Defining a Name for Camera
Capture = cv2.VideoCapture(0)
Capture1 = cv2.VideoCapture(1)

if __name__=='__main__':

    print 'Starting..'
    #Starting Capturing Frames and Display it
    while (True):

        success, frame = Capture.read()
        success1, frame1 = Capture1.read()

        t1 = cv2.getTickCount()
        
        result = cv2.addWeighted(frame, 0.2, frame1, 0.8, 0)

        if not (success or success1):
            print 'No Frames Available'
            break
        
        cv2.imshow("Frames", result)

        t2 = cv2.getTickCount()

        TimeTaken = (t2-t1)/cv2.getTickFrequency()
        print 'TimeTaken is : ', TimeTaken, '\r'
            
        #Check Whether q is Pressed or Not??
        key  = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    Capture.release()

    

    
