# Import necessary libraries.
import cv2
import datetime
import time

# Initialize the picamera.
Capture = cv2.VideoCapture(0)

# definig video parameters
VideoName = 'Motion_Output.avi'
saveVideo = False

# video creation object
Video = cv2.VideoWriter(VideoName, 0, 20.0, (640,480))

# Definig Parameters
showVideo =  True
camera_warmup_time = 0.5
delta_thresh = 5
min_area = 10000


# Allow the camera to warmup
print "[INFO] warming up..."
time.sleep(camera_warmup_time)

# capture frames from the camera
while (Capture.isOpened()):
    # Grab the frame
    success, frame = Capture.read()
    timestamp = datetime.datetime.now()
    #print timestamp
    text = "NO DETECTION"
    StartTime = cv2.getTickCount()

    #Check Whether frame is Availbale or not, if not then continue;
    if not success:
        print 'No Frames Available'
        continue
 
    # Convert it to grayscale, and blur it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
 
    # Copy Existing Frame
    frame_copy = gray.copy()

    #Capture another frame
    print 'Capturing Another Frame'
    success , frame = Capture.read()
    if not success:
        print 'No Copy Frames Available'
        break
    
    #Convert Captured frame into grayscale
    gray_copy = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_copy = cv2.GaussianBlur(gray_copy, (21, 21), 0)
 
    # Compute the difference between the current frame and running average
    frameDelta = cv2.absdiff(gray_copy, gray)

    # Threshold the delta image, dilate the thresholded image to fill in holes
    thresh = cv2.threshold(frameDelta, delta_thresh, 255,
                           cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=1) # Removes noise
    thresh = cv2.dilate(thresh, None, iterations=3)

    #Find Contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
 
    # loop over the contours
    for c in contours:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < min_area:
            continue
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.drawContours(frame, c, -1, (0, 0, 255), 3)
        text = "Motion"
        # save video
        if saveVideo is True:
            Video.write( frame )
 
    # draw the text and timestamp on the frame
    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    cv2.putText(frame, "Detection: {}".format(text), (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        0.35, (0, 255, 0), 1)

    EndTime = cv2.getTickCount()

    TimeTaken_to_Process = (EndTime - StartTime)/cv2.getTickFrequency()
    print '\rTime Taken to Process Frame is : ', TimeTaken_to_Process

    # check to see if the frames should be displayed to screen
    if showVideo is True:
        # display the security feed
        cv2.imshow("Feed", frame)
        cv2.imshow("thresh", thresh)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key is pressed, break from the loop
        if key == ord("q"):
            break
        
    else:
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key is pressed, break from the loop
        if key == ord("q"):
            break
 
# clear the stream in preparation for the next frame
Capture.release()
cv2.destroyAllWindows()
     
