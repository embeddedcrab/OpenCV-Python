'''
Feature-based image matching.

USAGE
  [--feature=<sift|surf|orb>[-flann]]

  --feature  - Feature to use. Can be sift, surf of orb. Append '-flann' to feature name
                to use Flann-based matcher instead bruteforce.
'''

import numpy as np
import cv2
from common import anorm, getsize

Capture = cv2.VideoCapture(0)

if ( not (Capture.isOpened()) ):
    exit()

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH    = 6
MIN_MATCH_COUNT = 15

# Load face cascades
face_cascade = cv2.CascadeClassifier('PATH\haarcascade_frontalface_alt.xml')

def init_feature(name):
    chunks = name.split('-')
    if chunks[0] == 'sift':
        detector = cv2.SIFT()
        norm = cv2.NORM_L2
    elif chunks[0] == 'surf':
        detector = cv2.SURF(800)	# Parameter values can be changed
        norm = cv2.NORM_L2
    elif chunks[0] == 'orb':
        detector = cv2.ORB(400)
        norm = cv2.NORM_HAMMING
    else:
        return None, None
    if 'flann' in chunks:
        if norm == cv2.NORM_L2:
            flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else:
            flann_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) #2
        matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    else:
        matcher = cv2.BFMatcher(norm)
    return detector, matcher


def filter_matches(kp1, kp2, matches, ratio = 0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs

def explore_match(win, img1, img2, kp_pairs, status = None, H = None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
    p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
            cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)
    vis0 = vis.copy()
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)

    cv2.imshow(win, vis)

if __name__ == '__main__':
    print __doc__

    import sys, getopt
    opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
    opts = dict(opts)
    feature_name = opts.get('--feature', 'sift-flann')		# Use different detectors, just change the name and/or append flann
    
	# Get base image for matching (taking a face here)
    ret, img1 = Capture.read()
    cv2.imshow('Fr1', img1)
    # convert train image into gray scale and find faces in it
    img1 = cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(img1, scaleFactor=1.3, minNeighbors=5, minSize=(20, 20))
    if len(faces):
        #Draw faces and find features
        for (x,y,w,h) in faces:
            # Draw rectangle on faces
            cv2.rectangle(img1,(x,y),(x+w,y+h),(0,0,255),2)
            # Crop faces from frame
            Cropped = img1[y:y+h, x:x+w]
            cv2.imshow('Cropped', Cropped)
            img1 = Cropped
    print 'Press any key to start matching..\n'
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
	# Initialize features for matching
    detector, matcher = init_feature(feature_name)
    if detector != None:
        print 'using', feature_name
    else:
        print 'unknown feature:', feature_name
        sys.exit(1)

    while( Capture.isOpened() ): 
        ret, frame = Capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5, minSize=(20, 20))
        #Draw faces and find features
        for (x,y,w,h) in faces:
            # Draw rectangle on faces
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),2)
            # Crop faces from frame
            Cropped = frame[y:y+h, x:x+w]
            img2 = Cropped

        cv2.imshow('Frame', frame)

        kp1, desc1 = detector.detectAndCompute(img1, None)
        kp2, desc2 = detector.detectAndCompute(img2, None)
        print 'img1 - %d features, img2 - %d features' % (len(kp1), len(kp2))

        if len(faces):

            def match_and_draw(win):
                raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
      
                p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
                if len(p1) >= MIN_MATCH_COUNT:
                    H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
                    #print '%d / %d  inliers/matched' % (np.sum(status), len(status))
                    print 'Matching..%d matches found\n\n' % len(p1)
                else:
                    H, status = None, None
                    #print '%d matches found, not enough for homography estimation' % len(p1)
                    print 'Not Matching, only %d mathces found not enough\n\n' % len(p1)

                vis = explore_match(win, img1, img2, kp_pairs, status, H)
                cv2.waitKey(1)

            match_and_draw('find_obj')
        
		# Break loop if q is pressed
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break;
        
    Capture.release()
    cv2.destroyAllWindows()
	# End
