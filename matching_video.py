import numpy as np
import cv2
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Matching Features in Live')

    ### Positional arguments    
    parser.add_argument('-c', '--cameraSource', default=0, help="Introduce number or camera path, default is 0 (default cam)")    

    args = vars(parser.parse_args())


    dim = (378, 504)
    img1 = cv2.imread('data/book.jpg',0)
    img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)            
    # Initiate ORB detector
    orb = cv2.ORB_create()


    # Start the video stream
    # 0 is default camera, change value for different input camera
    cap = cv2.VideoCapture(args["cameraSource"]) 
    cv2.namedWindow("sample", cv2.WINDOW_AUTOSIZE)
    
    while(True):
    
        # Get Frame
        ret, frame = cap.read()  

        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1,None)
        kp2, des2 = orb.detectAndCompute(frame,None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1,des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)        
        # Draw first 100 matches.
        img3 = cv2.drawMatches(img1,kp1,frame,kp2,matches[:100],frame, flags=2)
        # img3 = cv2.drawMatches(img1,kp1,frame,kp2,matches[:],frame, flags=2)
        cv2.imshow("sample", img3)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
    
