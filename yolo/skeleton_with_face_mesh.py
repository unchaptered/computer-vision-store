import cv2
import imageio
import numpy as np
import mediapipe as mp
from typing import List
from ultralytics import YOLO

FILE_NAME = 'IMG_0290.MOV'
IS_WRITABLE_MODE = False
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(1280))
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(720))

if IS_WRITABLE_MODE:
    capOutImagio = imageio.get_writer('OUT.mp4', fps=30)

MODEL = YOLO('yolov8n-pose.pt')
YOLO_KEYPNTS_LEN = 17

if not cap.isOpened():
    print('웹캠을 열 수 없습니다.')
    exit()


def cvtCenter(landmark: tuple):
    center: tuple = int(landmark[0]), int(landmark[1])
    return center


def isVerifiedLine(ketThres: List[float], boundThres: float, pt1: int, pt2: int):
    return ketThres[pt1] > boundThres and ketThres[pt2] > boundThres


def drawLine(frame: np.ndarray, keyThres: List[float], keyPt: List[tuple], boundThres: float, pt1: int, pt2: int):
    if isVerifiedLine(keyThres, boundThres, pt1, pt2):
        cv2.line(img=frame, pt1=cvtCenter(keyPt[pt1]), pt2=cvtCenter(
            keyPt[pt2]), color=(156, 164, 0), thickness=1)


def drawLines(frame: np.ndarray, keyThres: List[float], keyPt: List[tuple], boundThres: float, ptList: List[tuple]):
    for pt1, pt2 in ptList:
        drawLine(frame, keyThres, keyPt, boundThres, pt1, pt2)


ret, frame = cap.read()

PERSON_BOUNDARY = 0.7
MIN_WIDTH, MIN_HEIGHT = 0,              0
MAX_WIDTH, MAX_HEIGHT = frame.shape[1], frame.shape[0]

faceMesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    min_tracking_confidence=0.2,
    min_detection_confidence=0.2
)
mpDrawUtils = mp.solutions.drawing_utils

landmarkCashingStorage = None

while True:

    ret, frame = cap.read()

    try:
        if ret != True:
            break

        pResultList = MODEL.predict(
            source=frame, stream=True, verbose=False)
        for pResult in pResultList:

            pThresList = pResult.boxes.conf
            pAreaList = pResult.boxes.xyxy

            if len(pAreaList) == 0:
                continue

            for idx, pArea in enumerate(pAreaList):

                pThres = pThresList[idx]
                pArea = pArea

                if float(pThres) > PERSON_BOUNDARY:

                    MIN_GAP = 0.9
                    PLS_GAP = 1.1

                    x1, y1, x2, y2 = pArea
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    x1, y1, x2, y2 = int(
                        x1*MIN_GAP), int(y1*MIN_GAP), int(x2*PLS_GAP), int(y2*PLS_GAP)
                    x1, y1, x2, y2 = max(x1, MIN_WIDTH), max(
                        y1, MIN_HEIGHT), min(x2, MAX_WIDTH), min(y2, MAX_HEIGHT)

                    cropPArea = frame.copy()[y1:y2, x1:x2]

                    cropPGrayArea = cv2.cvtColor(
                        cropPArea, cv2.COLOR_RGB2GRAY)

                    faceMeshResult = faceMesh.process(cropPArea)
                    if faceMeshResult.multi_face_landmarks:
                        landmarkCashingStorage = faceMeshResult.multi_face_landmarks

                    if landmarkCashingStorage is not None:
                        for landmark in landmarkCashingStorage:
                            mpDrawUtils.draw_landmarks(
                                image=cropPArea,
                                landmark_list=landmark,
                                # connections=[(0, 1), (1, 2)],
                                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=mpDrawUtils.DrawingSpec(
                                    color=(0, 0, 255, 0), circle_radius=0, thickness=1),
                                connection_drawing_spec=mpDrawUtils.DrawingSpec(
                                    color=(255, 255, 255, 25), thickness=1)
                            )

                    frame[y1:y2, x1:x2] = cropPArea

                    keyThres = pResult.keypoints.conf[0]
                    boundThres = 0.3
                    keyPt = pResult.keypoints.xy[0]

                    exceptKeyPtList = [13, 14, 15, 16]
                    # exceptKeyPtList = [15, 16]
                    for idx, landmark in enumerate(keyPt):
                        if idx not in exceptKeyPtList:
                            center: tuple = cvtCenter(landmark=landmark)
                            if keyThres[idx] > boundThres:
                                cv2.circle(img=frame, center=center,
                                           radius=1, color=(250, 233, 86), thickness=2)

                    # 얼굴
                    drawLines(frame, keyThres, keyPt, boundThres, [
                        # 얼굴
                        (0, 1),    (1, 3),    (0, 2),    (2, 4),

                        # 상체
                        (5, 6),    (6, 12),   (12, 11),  (11, 5),

                        # 왼쪽 방향 팔           # 오른 방향 팔
                        (6, 8),    (8, 10),   (5, 7),    (7, 9),

                        # 왼쪽 방향 다리         # 오른 방향 다리
                        # (12, 14),             (11, 13),
                        # (12, 14),  (14, 16),  (11, 13),  (13, 15)
                    ])

        if IS_WRITABLE_MODE:
            fmt_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            capOutImagio.append_data(fmt_frame)
        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(e)
        cv2.imshow("Webcam", frame)

# 리소스를 해제합니다.
cap.release()
if IS_WRITABLE_MODE:
    capOutImagio.close()
cv2.destroyAllWindows()
