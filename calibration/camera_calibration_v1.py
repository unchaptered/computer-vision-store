import cv2
import numpy as np

CHECKERBOARD = (6, 9)
objpoints = []
imgpoints = []

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                          0:CHECKERBOARD[1]].T.reshape(-1, 2)


def calibrate_and_correct(img, mtx, dist):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h))
    corrected_img = cv2.undistort(img, mtx, dist, None, newcameramtx)
    return corrected_img


cap = cv2.VideoCapture(1)  # Logitech BRIO 카메라 연결

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret:
        cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, ret)
        objpoints.append(objp)
        imgpoints.append(corners)

    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1)
    if key == 13:  # Enter key
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
        corrected_frame = calibrate_and_correct(frame, mtx, dist)
        cv2.imshow('Corrected Frame', corrected_frame)
    elif key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
