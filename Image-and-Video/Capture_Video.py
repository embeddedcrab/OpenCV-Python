import cv2
import datetime

VideoName = 'Capture_Output.avi'
SaveVideo = True

Capture = cv2.VideoCapture(0)

Video = cv2.VideoWriter(VideoName, 0, 20.0, (640,480))

while(Capture.isOpened()):
    success, Image = Capture.read()

    if success != True:
        print 'No Frames'
        break

    if SaveVideo is True:
        Video.write(Image)

    cv2.imshow('VideoFrames',Image)

    key = (cv2.waitKey(1) & 0xFF)
    if key == ord("q"):
        print 'Switching OFF..!!'
        break

Capture.release()
Video.release()
cv2.destroyAllWindows()

        

            
                        
