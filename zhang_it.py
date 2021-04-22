import sys
import os
import cv2
import numpy as np

IMAGE_POINTS = np.int0([(0, 0), (0, 1), (1, 0), (1, 1)])
CIRCULAR_POINTS = np.array([[1, 1j, 0], [1, -1j, 0]])

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

def get_warped_circular(img):

    corners2 = IMAGE_POINTS.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    corners1 = cv2.goodFeaturesToTrack(gray, 4, 0.01, 10)
    corners1 = np.int0(corners1)

    corners1 = sort_sad(corners1)

    homo, _ = cv2.findHomography(corners1, corners2)

    circular = CIRCULAR_POINTS.copy().transpose()
    circular_warped = np.matmul(homo, circular)

    return warped_circular


def calculate_conic(points):
    x = points[:, 0]
    y = points[:, 1]

    Mat = np.vstack([x ** 2, x * y, y ** 2, x, y]).T
    fullSolution = np.linalg.lstsq(Mat, np.ones(x.size))
    (a, b, c, d, e) = fullSolution[0]
    ret = np.array([[a, b / 2, d / 2],
           [b / 2, c, e / 2],
           [d / 2, e / 2, 1]])
    return ret


if __name__ == "__main__":
    try:
        images_path = sys.argv[1]
        if not os.path.isdir(images_path):
            raise Exception("Please create directory data/ and place images inside.")
    except Exception:
        raise Exception("GIVE ME IMAGES!!")

    images = os.listdir(images_path)
    if not len(images) == 3:
        print("The pics I got were: \n")
        print(images)
        raise Exception("Didn't get 3 pictures from " + images_path)

    warped_circular = []
    for image in images:
        with cv2.imread(image) as im:
            warped_circular.append(get_warped_circular(im))

    conic = calculate_conic(warped_circular)
    camera_mat = np.linalg.cholesky(conic)
    # warped = cv2.warpPerspective(image, homo, (1, 1))
    #
    # cv2.imshow('warped', warped)

    # for corner in corners1:
    #     print(corner)
    #     x, y = corner.ravel()
    #     cv2.circle(image, (x, y), 3, 255, -1)
    #
    # cv2.imshow('Corner', image)

    # cv2.waitKey(0)
    #
    # cv2.destroyAllWindows()