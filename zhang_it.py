import sys
import os
import cv2
import numpy as np

def sort_sad(corners):
    corner_sums = []
    for corner in corners:
        sum = corner[0, 0] + corner[0, 1]
        corner_sums.append(sum)

    max_idx = np.argmax(corner_sums)
    min_idx = np.argmin(corner_sums)
    sorted = [[], [], [], []]
    for idx, corner in enumerate(corners):
        if idx == max_idx:
            sorted[3] = corner[0]
        elif idx == min_idx:
            sorted[0] = corner[0]
        elif corner[0, 0] > corner[0, 1]:
            sorted[2] = corner[0]
        else:
            sorted[1] = corner[0]

    return np.int0(sorted)


if __name__ == "__main__":
    # try:
    #     images_path = sys.argv[1]
    #     if not os.path.isdir(images_path):
    #         raise Exception("Please create directory data/ and place images inside.")
    # except Exception:
    #     raise Exception("GIVE ME IMAGES!!")

    # images = os.listdir(images_path)
    corners2 = np.int0([(0, 0), (0, 100), (100, 0), (100, 100)])
    image = cv2.imread("square.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    corners1 = cv2.goodFeaturesToTrack(gray, 4, 0.01, 10)
    corners1 = np.int0(corners1)

    # print(corners1)

    corners1 = sort_sad(corners1)

    # print(corners1)
    # exit()

    homo, _ = cv2.findHomography(corners1, corners2)

    warped = cv2.warpPerspective(image, homo, (100, 100))

    cv2.imshow('warped', warped)

    # for corner in corners1:
    #     print(corner)
    #     x, y = corner.ravel()
    #     cv2.circle(image, (x, y), 3, 255, -1)
    #
    # cv2.imshow('Corner', image)

    cv2.waitKey(0)

    cv2.destroyAllWindows()