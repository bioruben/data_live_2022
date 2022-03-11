import cv2
import argparse



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Matching Features in Live')

    ### Positional arguments    
    parser.add_argument('-c', '--cameraSource', default=0, help="Introduce number or camera path, default is 0 (default cam)")
    parser.add_argument('-s', '--sizeWindow', default=5, type=int, help="Define size of Gaussian Filter")
    parser.add_argument('-g', '--flagGaussian', default=False, help="Define size of Gaussian Filter")

    args = vars(parser.parse_args())

    size_window = args["sizeWindow"]
    flag_gaussian = args["flagGaussian"]

    cap = cv2.VideoCapture(args["cameraSource"])
    while cap.isOpened():

        #BGR image feed from camera
        ret, img = cap.read()
        #BGR to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if flag_gaussian:
            cv2.GaussianBlur(img_gray, (size_window,size_window), 0, img_gray)        
        edges = cv2.Canny(img_gray,10,20,apertureSize = 3)

        cv2.imshow("Faces found", edges)

        k = cv2.waitKey(10)
        if k==27:
            break

    cap.release()
    cv2.destroyAllWindows()



