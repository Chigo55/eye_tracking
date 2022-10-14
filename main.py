import cv2 as cv
import module as m
import os

def eye_tracking(image, magnificate=1, threshold=70):
    # image: 사진 이미지 주소, magnificate: 사진 확대 배율(기본 1배), threshold: 임계값 처리시 허용 임계값(배율 1당 threshold +- 10이 적당)

    # 이미즈의 크기를 읽어온다.
    height, width = image.shape[:2]

    # 읽어 들인 이미지의 크기를 이용하여 사진의 3배 확대 한다.
    image = cv.resize(image, (int(width * 3), int(height * 3)), cv.INTER_AREA)

    # 확대한 이미지를 회색 이미지로 변환한다.
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 사진에서 얼굴 부분을 감지하여 표시한다. -> clone, faces = m.faceDetector(image, gray_image, draw=True)
    clone, faces = m.faceDetector(image, gray_image)

    # 감지한 얼굴을 이용하여 랜드마크를 찾고 표시한다. -> clone, point_lists = m.faceLandmarkDetector(image, gray_image, faces, Draw=True)
    clone, point_lists = m.faceLandmarkDetector(image, gray_image, faces, Draw=False)

    # 랜드마크중 왼쪽 눈과 오른쪽 눈에 해당하는 랜드마크를 따로 가져온다.
    for point_list in point_lists:
        right_eye_point = point_list[36:42]
        left_eye_point = point_list[42:48]

        # 따로 가져온 랜드마크를 이용하여 눈동자를 찾는다.
        image, right_x_per, right_y_per = m.irisPosition(image, gray_image, right_eye_point, threshold)
        image, left_x_per, left_y_per = m.irisPosition(image, gray_image, left_eye_point, threshold)

    # 찾은 눈동자의 좌표를 출력한다.
    print("Right: " + "X" + "-" + str(right_x_per) + "%", "Y" + "-" + str(right_y_per) + "%")
    print("Left: " + "X" + "-" + str(left_x_per) + "%", "Y" + "-" + str(left_y_per) + "%")

    # 모든 과정이 끝난 사진의 크기를 다시 읽어온다.
    height, width = image.shape[:2]

    # 다시 읽어온 사진의 크기를 설정한 배율에 따라 크기를 변경한다.
    image = cv.resize(image, (int(width / 3) * magnificate, int(height / 3) * magnificate), cv.INTER_AREA)


    return image

# 여러장의 사진을 활용 할 때
img_path = "C:/Users/user/PycharmProjects/pythonProject/picture/"
image_list = os.listdir(img_path)
for data in image_list:
    print(data)
    img_data = cv.imread(img_path + data)
    image = eye_tracking(img_data, magnificate=2)

    cv.imshow(data, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

# 한장의 사진을 활용 할 때
# image = cv.imread("C:/Users/user/PycharmProjects/pythonProject/picture/2022.10.12 - 16.19.4.png")
# eye_tracking(image)
# cv.imshow("data", image)
# cv.waitKey(0)
# cv.destroyAllWindows()