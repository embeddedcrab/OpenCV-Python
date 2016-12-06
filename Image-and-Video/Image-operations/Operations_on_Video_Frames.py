#Script to perform operations on Video Frames

#Importing Necessary Libraries
import cv2
import datetime

#Variable to Access Camera
Capture = cv2.VideoCapture(0)

#Flag to save Images
Save_Image = False

#Names for Images
ImageNumber = 1
GrayImageNumber = 1
BlurImageNumber = 1


if __name__=='__main__':

    #Get Current Date and Time and Display 
    time = datetime.datetime.now()
    print time

    while (Capture.isOpened()):
        
        #Read an Image
        success, Image = Capture.read()

        if success is not True:
            print 'No farmes Available'
            break

        #Change Image into Gray-scale
        GrayImage = cv2.cvtColor(Image, cv2.COLOR_RGB2GRAY)
        #Blur Image
        BlurImage = cv2.GaussianBlur(Image, (21, 21), 0)

        #Save Image with Same or Different Format
        if Save_Image == True:
            cv2.imwrite('Image' + str(ImageNumber) + '.png', Image)
            cv2.imwrite('Gray_Image' + str(GrayImageNumber) + '.png', GrayImage)
            cv2.imwrite('Blur_Image' + str(BlurImageNumber) + '.png', BlurImage)
            #Increment Name of Image Numbers
            ImageNumber = ImageNumber + 1
            GrayImageNumber = GrayImageNumber + 1
            BlurImageNumber = BlurImageNumber + 1

        #Show Image
        cv2.imshow('Image', Image)
        cv2.imshow('Gray Image', GrayImage)
        cv2.imshow('Blur Image', BlurImage)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print 'Switching OFF'
            break

#Wait for Key input then Clear All Windows
Capture.release()
cv2.destroyAllWindows()
