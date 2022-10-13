import cv2 as cv
import module as m
import os

def eye_tracking(image, magnificate = 3):
    height, width = image.shape[:2]

    image = cv.resize(image, (int(width*magnificate), int(height*magnificate)), cv.INTER_AREA)

    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    clone, faces = m.faceDetector(image, gray_image)

    clone, point_lists = m.faceLandmarkDetector(image, gray_image, faces, Draw=False)

    for point_list in point_lists:
        right_eye_point = point_list[36:42]
        left_eye_point = point_list[42:48]

        image, right_x_per, right_y_per = m.irisPosition(image, gray_image, right_eye_point)
        image, left_x_per, left_y_per = m.irisPosition(image, gray_image, left_eye_point)

    print("Right: " + "X" + "-" + str(right_x_per) + "%", "Y" + "-" + str(right_y_per) + "%")
    print("Left: " + "X" + "-" + str(left_x_per) + "%", "Y" + "-" + str(left_y_per) + "%")

    return

img_path = "C:/Users/user/PycharmProjects/pythonProject/picture/"
image_list = os.listdir(img_path)
for data in image_list:
    print(data)
    img_data = cv.imread(img_path + data)
    image = eye_tracking(img_data)

    cv.imshow("", image)
    cv.waitKey(0)

"""image = cv.imread("C:/Users/user/PycharmProjects/pythonProject/picture/images8.jfif")
eye_tracking(image)"""