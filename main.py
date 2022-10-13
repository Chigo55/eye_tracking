import cv2 as cv
import test_module2 as m
import os

def eye_tracking(image):
    height, width = image.shape[:2]

    image = cv.resize(image, (int(width*3), int(height*3)), cv.INTER_AREA)

    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    clone, faces = m.faceDetector(image, gray_image)

    clone, point_lists = m.faceLandmarkDetector(image, gray_image, faces, Draw=False)

    for point_list in point_lists:
        right_eye_point = point_list[36:42]
        left_eye_point = point_list[42:48]

        r_mask = m.irisPosition(image, gray_image, right_eye_point)
        l_mask = m.irisPosition(image, gray_image, left_eye_point)

    cv.imshow("Picture", clone)
    cv.waitKey(0)

img_path = "C:/Users/user/PycharmProjects/pythonProject/picture/"
image_list = os.listdir(img_path)
for data in image_list:
    print(data)
    img_data = cv.imread(img_path + data)
    eye_tracking(img_data)

"""image = cv.imread("C:/Users/user/PycharmProjects/pythonProject/picture/images8.jfif")
eye_tracking(image)"""