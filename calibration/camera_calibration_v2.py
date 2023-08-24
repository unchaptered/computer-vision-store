import cv2
import numpy as np

from typing import Tuple

CHECKER_BOARD = (6, 9)
WIDTH = 1280
HEIGHT = 720
RESOLUTION = (WIDTH, HEIGHT)


def findChessboardCorners(gryFrm: np.ndarray) -> Tuple[any]:
    global CHECKER_BOARD

    ret, corners = cv2.findChessboardCorners(
        gryFrm,
        CHECKER_BOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH    # 이미지 이진화 시, 적응형 임계값 사용
        + cv2.CALIB_CB_FAST_CHECK       # CHECKER_BOADR 탐색 시, 없을 시 즉시 종료
        + cv2.CALIB_CB_NORMALIZE_IMAGE  # 이미지 명함 정규화
    )

    return ret, corners


def calibrateCamera(gryFrm: np.ndarray) -> Tuple[any]:
    global _3DPointList, _2DPointList

    repojError, intrisicMatrix, distCoeffs, rVectors, tVectors = cv2.calibrateCamera(
        _3DPointList,           # 3D 좌표
        _2DPointList,           # 2D 좌표
        gryFrm.shape[::-1],     # 행렬 규격을 전치 Tuple[A, B] -> Tuple[B, A]
        None,
        None
    )

    return repojError, intrisicMatrix, distCoeffs, rVectors, tVectors


def getOptimalNewCameraMatrix(intrisicMatrix: any, distCoeffs: any):
    global RESOLUTION

    newCamMatrix, roi = cv2.getOptimalNewCameraMatrix(
        intrisicMatrix,
        distCoeffs,
        RESOLUTION,
        1,
        RESOLUTION
    )
    return newCamMatrix, roi


_3DPointList = []  # 3D 좌표
_2DPointList = []  # 2D 좌표

# (1, 54, 3) 규격의 행렬으로 (x, y, z)로 구성됨
_3DPointVolume = np.zeros(
    (1, CHECKER_BOARD[0] * CHECKER_BOARD[1], 3),
    np.float32
)
_3DPointVolume[0, :, :2] = np.mgrid[
    0:CHECKER_BOARD[0],
    0:CHECKER_BOARD[1]
].T.reshape(-1, 2)

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)


while True:

    ret, frm = cap.read()

    if cap.isOpened() and ret == True:

        gryFrm = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        ret, corners = findChessboardCorners(gryFrm=gryFrm)

        if ret == True:
            cv2.drawChessboardCorners(frm, CHECKER_BOARD, corners, ret)

            _3DPointList.append(_3DPointVolume)
            _2DPointList.append(corners)

            repojError, intrisicMatrix, distCoeffs, rVectors, tVectors = calibrateCamera(
                gryFrm
            )
            newCamMetrix, roi = getOptimalNewCameraMatrix(
                intrisicMatrix,
                distCoeffs
            )
            correctImg = cv2.undistort(
                frm,
                intrisicMatrix,
                distCoeffs,
                None,
                newCamMetrix
            )

            x, y, w, h = roi
            croppedImg = correctImg.copy()[y:y+h, x:x+w]
            cv2.rectangle(correctImg, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow('fmr2', correctImg)
            cv2.imshow('fmr3', croppedImg)

            print('=====' * 3)
            print(roi)
            print(frm.shape)
            print(correctImg.shape)
            print('=====' * 3)

        cv2.imshow('frm', frm)
        cv2Key = cv2.waitKey(1)

        if cv2Key == 27:
            break

cap.release()
cv2.destroyAllWindows()
