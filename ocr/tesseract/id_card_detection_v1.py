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
        
        resizedImg = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        cv2.imshow('img', resizedImg)
        cv2.waitKey(0)
        
        grayImg = cv2.cvtColor(resizedImg, cv2.COLOR_BGR2GRAY)
        cv2.imshow('img', grayImg)
        cv2.waitKey(0)

        _, thresholdImg = cv2.threshold(grayImg, 127, 255, cv2.THRESH_BINARY)
        cv2.imshow('img', thresholdImg)
        cv2.waitKey(0)
        
        # [STEP 1] 최소단위 사각형 구하기
        contours, _ =cv2.findContours(thresholdImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        maxContour = max(contours, key=cv2.contourArea)
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
        minAreaImg = resizedImg[
            y
            : y + height,
            x 
            : x + width
        ]
        cv2.imshow('img', minAreaImg)
        cv2.waitKey(0)
        
        # # [STEP 2] 영역 구하기
        # edgedImg = cv2.Canny(minAreaImg.copy(), 75, 200)
        # contours, hierarchy = cv2.findContours(
        #     edgedImg,
        #     cv2.RETR_EXTERNAL,
        #     cv2.CHAIN_APPROX_SIMPLE
        # )
        # # contour 중 네 꼭짓점을 찾아내기 위해 가장 큰 contour를 선택합니다.
        # maxContour = max(contours, key=cv2.contourArea)

        # # contour를 근사화하여 꼭짓점 좌표 추출
        # epsilon = 0.04 * cv2.arcLength(maxContour, True)
        # approx = cv2.approxPolyDP(maxContour, epsilon, True)
        # for idx, point in enumerate(approx):
        #     x, y = point[0]
        #     cv2.circle(minAreaImg, (x, y), 5, (0 + idx * 50, 255 - idx * 30, 255 - idx * 30), 5)
        # minAreaImgHeight, minAreaImgWidth, channel = minAreaImg.shape
        
        # # [CODE] 배열 평탄화하고 정렬하는 안정성 코드 추가 필요
        # # copiedApprox = approx.copy()
        # # flattenApprox = copiedApprox.squeeze()
        # # print(flattenApprox)
        
        # leftTop = approx[0][0]
        # leftBottom = approx[1][0]
        # rightBotto = approx[2][0]
        # rightTop = approx[3][0]
        # srcPoints = np.array([ leftTop, rightTop, rightBotto, leftBottom ], dtype=np.float32)
        # dstPoints = np.array([
        #     [0, 0],
        #     [minAreaImgWidth - 1, 0],
        #     [minAreaImgWidth - 1, minAreaImgHeight - 1],
        #     [0, minAreaImgHeight - 1]
        # ], dtype=np.float32)
        
        # cv2.imshow('img', minAreaImg)
        # cv2.waitKey(0)
        
        # M = cv2.getPerspectiveTransform(srcPoints, dstPoints)
        # result = cv2.warpPerspective(minAreaImg, M, (minAreaImgWidth, minAreaImgHeight))
        
        # cv2.imshow('img', result)
        # cv2.waitKey(0)
        
        cv2.destroyAllWindows()

if __name__ == '__main__':
    
    import os
    
    absPath = os.getcwd()
    
    imgPathList = [
        'images/kevin_id_card_rotated_01.jpg',
        'images/kevin_id_card_rotated_02.jpg',
        'images/kevin_id_card_rotated_03.jpg',
        'images/kevin_id_card_rotated_04.jpg',
        'images/kevin_id_card_slanted_01.jpg',
        'images/kevin_id_card_slanted_02.jpg',
        'images/kevin_id_card_slanted_03.jpg',
        'images/kevin_id_card_slanted_04.jpg',
        'images/kevin_id_card.jpg',
    ]
    
    for imgPath in imgPathList:    
        totalPath = os.path.join(absPath, imgPath)
        detectIdCard(totalPath)