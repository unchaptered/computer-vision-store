import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from time import sleep
from math import ceil, sqrt

from typing import Tuple
from imutils.perspective import four_point_transform
from ultralytics import YOLO

def calcDistance(pt1: np.ndarray, pt2: np.ndarray):
    x1, y1 = pt1[0], pt1[1]
    x2, y2 = pt2[0], pt2[1]
    return sqrt((x2-x1)**2 + (y2-y1)**2)

def convertValidApprox(
    source: np.ndarray,
    approx: np.ndarray,
):
    
    squeezedApprox = np.squeeze(approx)
    
    sqApproxLen = len(squeezedApprox)
    isValidApprox = sqApproxLen == 4
    if not isValidApprox:
        return None
    colorList =[
        (255, 0, 0), # 파랑색
        (0, 255, 255), # 노랑색
        (0, 0, 255), # 빨강색
        (0, 255, 0), # 녹색
        (0, 200, 0), # 녹색
        (0, 150, 0), # 녹색
        (0, 100, 0), # 녹색
        (0, 50, 0), # 녹색
        (0, 50, 0), # 녹색
        (0, 50, 0), # 녹색
        (0, 50, 0), # 녹색
        (0, 50, 0), # 녹색
        (0, 50, 0), # 녹색
        (0, 50, 0), # 녹색
    ]
    
    col, row = 7, 4
    sqApproxDistances = np.empty((row, col), dtype=np.float64)
    for idx, sqApprox in enumerate(squeezedApprox):
        
        isLastIdx = idx == (sqApproxLen - 1)

        if isLastIdx:
            nowIdx = idx
            tarIdx = 0
            
            nowApprox = squeezedApprox[idx].astype(float)
            tarApprox = squeezedApprox[0].astype(float)
            distance = calcDistance(nowApprox, tarApprox)
            
            # For Debug
            print(distance, nowApprox, tarApprox)
            print(distance, nowApprox[0], nowApprox[1], tarApprox[0], tarApprox[1])
            print(distance + nowApprox[0] + nowApprox[1] + tarApprox[0] + tarApprox[1])
            print(type(distance), type(nowApprox[0]), type(nowApprox[1]), type(tarApprox[0]), type(tarApprox[1]))
        
            arr = np.array([distance, nowIdx, nowApprox[0], nowApprox[1], tarIdx, tarApprox[0], tarApprox[1]], dtype=np.float32)
            sqApproxDistances[idx] = arr
        
            continue
        
        nowIdx = idx
        tarIdx = idx + 1
        
        nowApprox = squeezedApprox[nowIdx].astype(float)
        tarApprox = squeezedApprox[tarIdx].astype(float)
        distance = calcDistance(nowApprox, tarApprox)
        # For Debug        
        print(distance, nowApprox, tarApprox)
        print(distance, nowApprox[0], nowApprox[1], tarApprox[0], tarApprox[1])
        print(distance + nowApprox[0] + nowApprox[1] + tarApprox[0] + tarApprox[1])
        print(type(distance), type(nowApprox[0]), type(nowApprox[1]), type(tarApprox[0]), type(tarApprox[1]))

        arr = np.array([distance, nowIdx, nowApprox[0], nowApprox[1], tarIdx, tarApprox[0], tarApprox[1]], dtype=np.float32)
        sqApproxDistances[idx] = arr
        
        continue
    
    sqApproxDistances = sqApproxDistances[sqApproxDistances[:, 0].argsort()][::-1]
    
    lineOne = sqApproxDistances[0]
    lineTwo = sqApproxDistances[1]
    
    leftTop, rightTop, leftBtm, rightBtm = None, None, None, None
    isLineOneTopLine = (lineOne[3] + lineOne[6]) < (lineTwo[3] + lineTwo[6])
    topLine   = lineOne if isLineOneTopLine else lineTwo
    btmLine  = lineOne if not isLineOneTopLine else lineTwo
    # print(topLine)
    # print(btmLine)
    
    leftTopIdx    = np.argmin([topLine[2], topLine[5]]) * 3 + 1
    rightTopIdx   = np.argmax([topLine[2], topLine[5]]) * 3 + 1
    # print(leftTopIdx)
    # print(rightTopIdx)

    leftBtmIdx    = np.argmin([btmLine[2], btmLine[5]]) * 3 + 1
    rightBtmIdx   = np.argmax([btmLine[2], btmLine[5]]) * 3 + 1
    # print(leftBtmIdx)
    # print(rightBtmIdx)

    leftTop       = topLine[leftTopIdx  : leftTopIdx +3]
    rightTop      = topLine[rightTopIdx : rightTopIdx+3]
    
    leftBtm       = btmLine[leftBtmIdx  : leftBtmIdx +3]
    rightBtm      = btmLine[rightBtmIdx : rightBtmIdx+3]
    
    # print(leftTop)
    # print(rightTop)
    # print(leftBtm)
    # print(rightBtm)
    # print('=' * 20)

    return np.array([leftTop, rightTop, leftBtm, rightBtm])
    # colorList =[
    #     (255, 0, 0), # 파랑색
    #     (0, 255, 255), # 노랑색
    #     (0, 0, 255), # 빨강색
    #     (0, 255, 0), # 녹색
    #     (0, 200, 0), # 녹색
    #     (0, 150, 0), # 녹색
    #     (0, 100, 0), # 녹색
    #     (0, 50, 0), # 녹색
    #     (0, 50, 0), # 녹색
    #     (0, 50, 0), # 녹색
    #     (0, 50, 0), # 녹색
    #     (0, 50, 0), # 녹색
    #     (0, 50, 0), # 녹색
    #     (0, 50, 0), # 녹색
    # ]
    # for idx, sApprox in enumerate(squeezedApprox):
    #     cv2.circle(
    #         img=source,
    #         center=(sApprox[0], sApprox[1]),
    #         radius=10,
    #         color=colorList[idx],
    #         thickness=3
    #     )
    
    # return approx

# 버그 1 : 이미지 원근 복구할때 비율이 정적이라서 이미지가 찌그러질 수 있음
# 버그 2 : 사람의 옷이 Contours로 찍히는 버그가 있음 
def detectIdCard(imgPath: str ='./images/kevin_id_card.jpg'):
    
    img = cv2.imread(imgPath)

    if img is None:
        print('이미지 없음')
    else:
        
        resizedImg = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        cv2.imshow(imgPath, resizedImg)
        cv2.waitKey(0)
        
        # [STEP 1] 최소단위 사각형 구하기
        grayImg = cv2.cvtColor(resizedImg, cv2.COLOR_BGR2GRAY)
        cv2.imshow(imgPath, grayImg)
        cv2.waitKey(0)

        _, thresholdImg = cv2.threshold(grayImg, 127, 255, cv2.THRESH_BINARY)
        cv2.imshow(imgPath, thresholdImg)
        cv2.waitKey(0)
        
        contours, _ = cv2.findContours(thresholdImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        cv2.imshow(imgPath, minAreaImgV1)
        cv2.waitKey(0)
        
        grayMinAreaImgV1 = grayImg = cv2.cvtColor(minAreaImgV1, cv2.COLOR_BGR2GRAY)
        
        # [STEP 2] 원근 복구
        edgedImg = cv2.Canny(grayMinAreaImgV1.copy(), 75, 200)
        contours, hierarchy = cv2.findContours(
            edgedImg,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
            # cv2.CHAIN_APPROX_SIMPLE
        )
        # contour 중 네 꼭짓점을 찾아내기 위해 가장 큰 contour를 선택합니다.
        maxContour = max(contours, key=cv2.contourArea)

        # contour를 근사화하여 꼭짓점 좌표 추출
        epsilon = 0.04 * cv2.arcLength(maxContour, True)
        approx = cv2.approxPolyDP(maxContour, epsilon, True)
        print(approx)
        approx = convertValidApprox(
            source=minAreaImgV1,
            approx=approx,
        )
        if approx is None:
            cv2.destroyAllWindows()
            return
        print(approx)
        colorList =[
            (255, 0, 0), # 파랑색
            (0, 255, 255), # 노랑색
            (0, 0, 255), # 빨강색
            (0, 255, 0), # 녹색
            (0, 200, 0), # 녹색
            (0, 150, 0), # 녹색
            (0, 100, 0), # 녹색
            (0, 50, 0), # 녹색
            (0, 50, 0), # 녹색
            (0, 50, 0), # 녹색
            (0, 50, 0), # 녹색
            (0, 50, 0), # 녹색
            (0, 50, 0), # 녹색
            (0, 50, 0), # 녹색
        ]
        print('yaho ✅✅')
        for idx, a in enumerate(approx):
            cv2.circle(
                img=minAreaImgV1,
                center=(int(a[1]), int(a[2])),
                radius=5,
                color=colorList[idx],
                thickness=3
            )
        print('yaho ✅✅😊')
        cv2.imshow(imgPath, minAreaImgV1)
        cv2.waitKey(0)
        print('yaho 👿')
        
        leftTop     = approx[0][1:]
        rightTop    = approx[1][1:]
        leftBtm     = approx[2][1:]
        rightBtm    = approx[3][1:]
        
        idStandardWidth  = 860
        idStandardHeight = 540
        
        srcPoints = np.array([ leftTop, rightTop, rightBtm, leftBtm ], dtype=np.float32)
        dstPoints = np.array([[0, 0], [idStandardWidth - 1, 0], [idStandardWidth - 1, idStandardHeight - 1], [0, idStandardHeight - 1]], dtype=np.float32)
        cv2.imshow(imgPath, minAreaImgV1)
        cv2.waitKey(0)
        
        M = cv2.getPerspectiveTransform(srcPoints, dstPoints)
        result = cv2.warpPerspective(minAreaImgV1, M, (idStandardWidth, idStandardHeight))
        
        print(result.shape)
        print(result.shape)
        print(result.shape)
        print(result.shape)
        
        cv2.imshow(imgPath, result)
        cv2.waitKey(0)
        
        grayResult = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        blurredFactor = 25
        blurredResult = cv2.medianBlur(grayResult, blurredFactor)
        reflectRemovedResult = cv2.absdiff(grayResult, blurredResult)
        
        cv2.imshow(imgPath, reflectRemovedResult)
        cv2.waitKey(0)
        
        # [STEP 4] 대조 강화
        hsvResult = reflectRemovedResult

        avgColor = np.mean(hsvResult)
        
        brightMask = hsvResult > avgColor
        brightFactor = 1.1
        hsvResult[brightMask] = np.clip(hsvResult[brightMask] * brightFactor, 0, 255)
        
        darkenMask = hsvResult < avgColor
        darkenFactor = 0.9
        hsvResult[darkenMask] = np.clip(hsvResult[darkenMask] * darkenFactor, 0, 255)
        
        result = cv2.bitwise_not(hsvResult)
        cv2.imshow(imgPath, result)
        cv2.waitKey(0)
        
        print('⭕⭕⭕⭕')
        print('⭕⭕⭕⭕')
        print('⭕⭕⭕⭕')
        print('⭕⭕⭕⭕')
        print('⭕⭕⭕⭕')
        
        # [STEP 5] Scale 강화
        # contrastFactor = 1.5
        # enhancedResult = cv2.convertScaleAbs(result, alpha=contrastFactor, beta=0)
        # cv2.imshow(imgPath, enhancedResult)
        # cv2.waitKey(0)
        
        print('❌❌❌❌')
        print('❌❌❌❌')
        print('❌❌❌❌')
        print('❌❌❌❌')
        print('❌❌❌❌')
        # [STEP 6] 배색팽창(Dialation) 강화
        thicknessFactor = 1.5
        _, binaryResult = cv2.threshold(result, np.mean(result) - 7, 255, cv2.THRESH_BINARY)
        cv2.imshow(imgPath, binaryResult)
        cv2.waitKey(0)
        dialationKernel = np.ones((int(thicknessFactor), int(thicknessFactor)), np.uint8)
        print('❗❗❗❗')
        print('❗❗❗❗')
        print('❗❗❗❗')
        print('❗❗❗❗')
        print('❗❗❗❗')
        dialationResult = cv2.dilate(binaryResult, dialationKernel, iterations=1)
        cv2.imshow(imgPath, dialationResult)
        cv2.waitKey(0)
        # [STEP 7] 노이즈 제거(Median Filter) 사용
        
        medianKernelSize = 3
        deNoisedResult = cv2.medianBlur(dialationResult, medianKernelSize)
        cv2.imshow(imgPath, deNoisedResult)
        cv2.waitKey(0)
        
        # COLOR 사진에 가능
        # hsvResult = cv2.cvtColor(reflectRemovedResult, cv2.COLOR_BGR2HSV)

        # avgColor = np.mean(hsvResult[:, :, 2])        
        
        # brightMask = hsvResult[:, :, 2] > avgColor
        # brightFactor = 1.1
        # hsvResult[:, :, 2][brightMask] = np.clip(hsvResult[:, :, 2][brightMask] * brightFactor, 0, 255)
        
        # darkenMask = hsvResult[:, :, 2] < avgColor
        # darkenFactor = 0.9
        # hsvResult[:, :, 2][darkenMask] = np.clip(hsvResult[:, :, 2][darkenMask] * darkenFactor, 0, 255)
        
        # result = cv2.cvtColor(hsvResult, cv2.COLOR_HSV2BGR)


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