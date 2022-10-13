import cv2 as cv
import dlib
import math
import numpy as np

import module

fonts = cv.FONT_HERSHEY_COMPLEX

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


def Position(ValueList):
    maxIndex = ValueList.index(max(ValueList))

    posEye = ""

    color = [WHITE, BLACK]

    if maxIndex == 0:
        posEye = "Right"
        color = [YELLOW, BLACK]
    elif maxIndex == 1:
        posEye = "Center"
        color = [BLACK, MAGENTA]
    elif maxIndex == 2:
        posEye = "Left"
        color = [LIGHT_CYAN, BLACK]
    else:
        posEye = "Eye Closed"
        color = [BLACK, WHITE]

    return posEye, color


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


def irisPosition(image, eye_points):
    # 컬러 이미지 그레이 스케일 이미지로 변환
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 그레이 스케일 이미지의 디멘션 get
    dim = gray.shape

    # 마스크 생성
    mask = np.zeros(dim, dtype=np.uint8)

    # 눈의 랜드마크 넘버링을 numpy의 array를 통해 변환
    polly_points = np.array(eye_points, dtype=np.int32)

    # 눈의 랜드마크를 흰색으로 채운다
    cv.fillPoly(mask, [polly_points], 255)

    # Bitwise and 오퍼레이터를 이용하여 마스크에 검은색으로 그레이 이미지 생성
    eye_image = cv.bitwise_and(gray, gray, mask=mask)

    # 눈 이미지의 최소 지점과 최대 지점의 좌표를 구한다
    maxX = (max(eye_points, key=lambda item: item[0]))[0]
    minX = (min(eye_points, key=lambda item: item[0]))[0]
    maxY = (max(eye_points, key=lambda item: item[1]))[1]
    minY = (min(eye_points, key=lambda item: item[1]))[1]

    # 검은색의 마스크를 씌운 그레이 이미지의 마스크를 흰색으로 변환
    eye_image[mask == 0] = 255

    eye_image   = eye_image[minY:maxY, minX:maxX]

    # 사진의 잡음 제거를 위해 가우시안 블러를 이용하여 블러 처리
    eye_image = cv.GaussianBlur(eye_image, (0, 0), 1)

    # 전역 임계처리를 이용하여 이미지의 이진화 및 임계처리
    _, threshold_eye = cv.threshold(eye_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # 이진화된 이미지를 이용하여 이지미에서 눈동자로 인식되는 외각선 검출
    contour = contours(threshold_eye)

    # 인식된 외각선 중 가장 큰 1개만을 가져옴
    cnt = contour[0]

    # convexhull을 이용하여 외각선 근사 처리
    hull = cv.convexHull(cnt)

    # 근사처리한 외각선을 그린다
    cv.drawContours(image, [hull], -1, BLUE, 2)

    # 외각선의 무게중심을 구한다
    moment = cv.moments(hull)

    # 외각선의 무게중심 좌표를 가져와 사진의 좌표값을 구한다
    cx = int(moment["m10"] / moment["m00"])
    cy = int(moment["m01"] / moment["m00"])

    # 눈동자의 좌표 출력
    print("눈동자 중심점", (cx, cy))



    return mask

def contours(threshold_image):
    # 이진화된 이미지에서 눈동자의 윤곽을 찾는다
    contours, _ = cv.findContours(threshold_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # 찾은 윤곽을 정렬하여 눈동자의 위치 좌표 검출
    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=False)

    return contours