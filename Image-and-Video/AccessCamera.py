#!/usr/bin/env python

import cv2
import numpy as np
#import video

if __name__=='__main__':

    import sys
    try:
        cam = sys.argv[1]
    except:
        cam = 0

    #capture = video.create_capture(cam)
    capture = cv2.VideoCapture(cam)
    success, frame = capture.read()

    if not success:
        print 'No Fframes Available'

    cv2.imshow('Frame', frame)
    cv2.waitKey(0)
    capture.release()
    cv2.destroyAllWindows()
    
