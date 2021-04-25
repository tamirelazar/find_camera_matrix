import re
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
    mx = corners[max_idx]
    mn = corners[min_idx]
    sort = [mn]
    middle = []
    for idx, corner in enumerate(corners):
        if idx == max_idx or idx == min_idx:
            pass
        else:
            middle.append(corner)
    if middle[0][0, 1] > middle[1][0, 1]:
        sort.append(middle[1])
        sort.append(middle[0])
    else:
        sort.append(middle[0])
        sort.append(middle[1])
    sort.append(mx)

    return sort


def get_warped_circular(img):
    corners2 = IMAGE_POINTS.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    corners1 = cv2.goodFeaturesToTrack(gray, 4, 0.01, 10)
    corners1 = np.int0(corners1)

    corners1 = np.float32(sort_sad(corners1))
    homography, _ = cv2.findHomography(corners2, corners1, cv2.RANSAC, 5.0)

    circular = CIRCULAR_POINTS.copy().transpose()
    circular_warped_a = np.matmul(homography, circular[:, 0])
    circular_warped_b = np.matmul(homography, circular[:, 1])

    return [circular_warped_a, circular_warped_b]


def calculate_conic(points):
    """
    Given six points, calculates the conic that fits all of them.
    :param points: a points array of size (6,2)
    :return: matrix representation of the fitted conic
    """
    calc_conic = np.zeros(shape=(6, 6), dtype=complex)
    for i in range(6):
        point = points[i]
        calc_conic[i] = (point[0]**2, point[0]*point[1],
                         point[1]**2, point[0]*point[2],
                         point[1]*point[2], point[2]**2)

    u, s, vh = np.linalg.svd(calc_conic)
    (a, b, c, d, e, f) = vh[-1]

    ret = np.array([[a, b / 2, d / 2],
           [b / 2, c, e / 2],
           [d / 2, e / 2, f]])
    return ret


if __name__ == "__main__":
    try:
        images_path = sys.argv[1]
        if not os.path.isdir(images_path):
            raise Exception("Please create directory data/ and place images inside.")
    except Exception:
        raise Exception("GIVE ME IMAGES!!")

    images = [f for f in os.listdir(images_path) if re.match(r'.*\.(jpeg|jpg|png)', f)]
    if not len(images) == 3:
        print("The pics I got were: \n")
        print(images)
        raise Exception("Didn't get 3 pictures from " + images_path)

    warped_circular = []
    for image in images:
        im = cv2.imread(images_path + "/" + image)
        warped_circular += (get_warped_circular(im))

    warped_circular = np.array(warped_circular)
    conic = calculate_conic(warped_circular)

    camera_mat = np.linalg.cholesky(conic)
    camera_mat = np.real(camera_mat.T / camera_mat[2, 2])
    print("\nThe camera calibration matrix calculated from the pictures in " + images_path + "/ is:")
    print(camera_mat)
    print("\nCalibrate responsibly!")
