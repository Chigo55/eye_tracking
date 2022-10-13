import cv2 as cv
import numpy as np
import dlib
import math

# variables
fonts = cv.FONT_HERSHEY_COMPLEX

# colors
YELLOW = (0, 247, 255)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 242)
GOLDEN = (32, 218, 165)
LIGHT_BLUE = (255, 9, 2)
PURPLE = (128, 0, 128)
CHOCOLATE = (30, 105, 210)
PINK = (147, 20, 255)
ORANGE = (0, 69, 255)
GREEN = (0, 255, 0)
LIGHT_GREEN = (0, 255, 13)
LIGHT_CYAN = (255, 204, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_RED = (2, 53, 255)

detectFace = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# function


def midpoint(points1, points2):
    x, y = points1
    x1, y1 = points2

    # 두 점의 가운데 점 계산
    x_out = int((x + x1) / 2)
    y_out = int((y + y1) / 2)

    return (x_out, y_out)


def euclideanDistance(points1, points2):
    x, y = points1
    x1, y1 = points2

    # 유클리디안 거리 계산
    euclidean_distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)

    return euclidean_distance

# creating face detector function


def faceDetector(image, gray, Draw=True):
    cord_face1 = (0, 0)
    cord_face2 = (0, 0)

    # 얼굴 디텍터로 얼굴를 찾는다
    faces = detectFace(gray)

    face = None
    # 루프 동안 모든 얼굴을 찾는다
    for face in faces:
        cord_face1 = (face.left(), face.top())
        cord_face2 = (face.right(), face.bottom())

        # Draw가 True이면 사각형을 그린다
        if Draw == True:
            cv.rectangle(image, cord_face1, cord_face2, GREEN, 1)

    return image, faces


def faceLandmarkDetector(image, gray, faces, Draw=True):
    point_lists = []

    for face in faces:
        # 얼굴의 랜드마크를 선행학습된 모듈을 통해 찾는다
        # shape_predictor_68_face_landmarks
        landmarks = predictor(gray, face)

        # 랜드마크의 위치를 넘버링한 것을 리스트에 저장
        point_list = []

        for n in range(0, 68):
            # 랜드마크의 각 넘버링을 x, y축으로 분할하여 저장
            point = (landmarks.part(n).x, landmarks.part(n).y)
            point_list.append(point)

            # Draw가 True이면 랜드마크를 그린다
            if Draw == True:
                cv.circle(image, point, 2, ORANGE, -1)

        point_lists.append(point_list)

    return image, point_lists

# Blink detector function.


def blinkDetector(eye_point):
    # 한쪽 눈의 랜드마크를 상하로 분할
    top = eye_point[1:3]
    bottom = eye_point[4:6]

    # 상하로 분할한 것에 중앙점 표시
    top_mid = midpoint(top[0], top[1])
    bottom_mid = midpoint(bottom[0], bottom[1])

    # 상하로 분할한 점의 거리를 계산
    Vertical_distance = euclideanDistance(top_mid, bottom_mid)
    Horizontal_distance = euclideanDistance(eye_point[0], eye_point[1])

    # 계산한 거리값을 이용하여 눈의 깜빡임을 인식
    blinkRatio = (Horizontal_distance / Vertical_distance)

    return blinkRatio, top_mid, bottom_mid

# Eyes Tracking function.


def irisPosition(image, gray, eyePoints):
    # getting dimensions of image
    dim = gray.shape
    # creating mask .
    mask = np.zeros(dim, dtype=np.uint8)

    # converting eyePoints into Numpy arrays.
    PollyPoints = np.array(eyePoints, dtype=np.int32)
    # Filling the Eyes portion with WHITE color.
    cv.fillPoly(mask, [PollyPoints], 255)

    # Writing gray image where color is White  in the mask using Bitwise and operator.
    eye_image = cv.bitwise_and(gray, gray, mask=mask)

    # getting the max and min points of eye inorder to crop the eyes from Eye image .

    maxX = (max(eyePoints, key=lambda item: item[0]))[0]
    minX = (min(eyePoints, key=lambda item: item[0]))[0]
    maxY = (max(eyePoints, key=lambda item: item[1]))[1]
    minY = (min(eyePoints, key=lambda item: item[1]))[1]

    # other then eye area will black, making it white
    eye_image[mask == 0] = 255
    # cv.imshow("1", eye_image)

    # 적응형 임계값 처리를 이용하여 이미지의 이진화 및 자동화 임계값 계산
    eye_image = cv.GaussianBlur(eye_image, (0, 0), 1)

    # threshold_eye = cv.adaptiveThreshold(eye_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 51, 2)
    _, threshold_eye = cv.threshold(eye_image, 45, 255, cv.THRESH_BINARY_INV)

    # 이진화된 이미지를 이용하여 이지미에서 눈동자의 위치 검출
    contour = contours(threshold_eye)

    # 이미지의 가로, 세로 길이를 가져옴
    rows, cols = eye_image.shape

    # 검출한 눈동자의 위치를 이용하여 인식한 눈동자의 외각선과 위치 좌표 줄력
    cnt = contour[0]
    hull = cv.convexHull(cnt)
    cv.drawContours(image, [hull], -1, BLUE, 2)

    moment = cv.moments(hull)
    cx = int(moment["m10"] / moment["m00"])
    cy = int(moment["m01"] / moment["m00"])
    print("눈동자 중심점", (cx, cy))


    return image

def contours(threshold_image):
    # 이진화된 이미지에서 눈동자의 윤곽을 찾는다
    contours, _ = cv.findContours(threshold_image, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_KCOS)

    # 찾은 윤곽을 정렬하여 눈동자의 위치 좌표 검출
    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)

    return contours