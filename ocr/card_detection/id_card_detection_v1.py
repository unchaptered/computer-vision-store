import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from time import sleep
from math import ceil
from imutils.perspective import four_point_transform

def detectIdCard(imgPath: str ='./images/kevin_id_card.jpg'):
    
    img = cv2.imread(imgPath)

    if img is None:
        print('이미지 없음')
    else:
        
        # [STEP 1] Resize Image more smaller than origin image.
        resizedImg = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        cv2.imshow('img', resizedImg)
        cv2.waitKey(0)
        
        # [STEP 2] Convert to grayscale image.
        grayImg = cv2.cvtColor(resizedImg, cv2.COLOR_BGR2GRAY)
        cv2.imshow('img', grayImg)
        cv2.waitKey(0)

        # [STEP 3] Convert to binary image.
        _, thresholdImg = cv2.threshold(grayImg, 127, 255, cv2.THRESH_BINARY)
        cv2.imshow('img', thresholdImg)
        cv2.waitKey(0)
        
        # [STEP 4] Find ID Card
        # [STEP 4-1] Find all contours
        contours, _ =cv2.findContours(thresholdImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # [STEP 4-2] Find max contour
        maxContour = max(contours, key=cv2.contourArea)
        
        # [STEP 4-3] GET Bounding Rect
            # [CASE A] Crop with padding
        x, y, width, height= cv2.boundingRect(maxContour)
        paddingX = ceil(width * 0.03)
        paddingY = ceil(height * 0.03)
        minAreaImgV1 = resizedImg[
            y - paddingY
            : y + paddingY + height,
            x - paddingX
            : x + paddingX + width
        ]
        cv2.imshow('img', minAreaImgV1)
        cv2.waitKey(0)
        
        # [CASE A] Crop without padding
        minAreaImg = resizedImg[
            y
            : y + height,
            x 
            : x + width
        ]
        cv2.imshow('img', minAreaImg)
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()

if __name__ == '__main__':
    
    import os
    
    absPath = os.getcwd()
    
    imgPathList = [
        'images/secure/kevin_id_card_rotated_01.jpg',
        'images/secure/kevin_id_card_rotated_02.jpg',
        'images/secure/kevin_id_card_rotated_03.jpg',
        'images/secure/kevin_id_card_rotated_04.jpg',
        'images/secure/kevin_id_card_slanted_01.jpg',
        'images/secure/kevin_id_card_slanted_02.jpg',
        'images/secure/kevin_id_card_slanted_03.jpg',
        'images/secure/kevin_id_card_slanted_04.jpg',
        'images/secure/kevin_id_card.jpg',
    ]
    
    for imgPath in imgPathList:    
        totalPath = os.path.join(absPath, imgPath)
        detectIdCard(totalPath)