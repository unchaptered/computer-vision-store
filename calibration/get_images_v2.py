# 마우스 좌클릭마다 이미지 저장

import cv2


def save_image(event, x, y, flags, param):
    global num, img
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.imwrite('images/img' + str(num) + '.png', img)
        print("image saved!")
        num += 1


cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

num = 0
img = None

cv2.namedWindow('Img')
cv2.setMouseCallback('Img', save_image, param=img)
while cap.isOpened():

    success, img = cap.read()
    if not success:
        break

    k = cv2.waitKey(5)

    if k == 27:
        break

    cv2.imshow('Img', img)

# Release and destroy all windows before termination
cap.release()
cv2.destroyAllWindows()
