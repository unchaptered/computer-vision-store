# 마우스 좌클릭마다 이미지 저장

import cv2
import numpy as np
from time import sleep


def save_image(event, x, y, flags, param):
    global num, savedImg
    if event == cv2.EVENT_LBUTTONDOWN:
        # sleep(3)

        cv2.imwrite('logitech_brio_1920_1080_v2_spair_s2/img' +
                    str(num) + '.png', savedImg)
        print("image saved!")
        num += 1


cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

num = 0
img = None
savedImg = None

chessboardSize = (9, 6)
frameSize = (1920, 1080)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0],
                       0:chessboardSize[1]].T.reshape(-1, 2)

size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.


cv2.namedWindow('Img')
cv2.setMouseCallback('Img', save_image, param=img)
while cap.isOpened():

    success, img = cap.read()
    savedImg = img.copy()
    if not success:
        break

    k = cv2.waitKey(5)

    if k == 27:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, chessboardSize, corners2, ret)

    resImg = cv2.resize(img, (int(1920*0.7), int(1080*0.7)))
    cv2.imshow('Img', resImg)

# Release and destroy all windows before termination
cap.release()
cv2.destroyAllWindows()
