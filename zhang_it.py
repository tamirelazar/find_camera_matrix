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

    # for corner in corners1:
    #     print(corner)
    #     x, y = corner.ravel()
    #     cv2.circle(img, (x, y), 3, 255, -1)
    #
    # cv2.imshow('Corner', img)
    #
    # cv2.waitKey(0)
    corners1 = np.float32(sort_sad(corners1))
    homo, _ = cv2.findHomography(corners1, corners2, cv2.RANSAC, 5.0)

    return homo

    circular = CIRCULAR_POINTS.copy().transpose()
    circular_warped_a = np.matmul(homo, circular[:, 0])
    circular_warped_b = np.matmul(homo, circular[:, 1])

    return [circular_warped_a, circular_warped_b]


def calculate_conic(points):
    x = points[:, 0]
    y = points[:, 1]

    Mat = np.vstack([x ** 2, x * y, y ** 2, x, y]).T
    fullSolution = np.linalg.lstsq(Mat, np.ones(x.size), rcond=None)
    (a, b, c, d, e) = fullSolution[0]

    ret = np.array([[a, b / 2, d / 2],
           [b / 2, c, e / 2],
           [d / 2, e / 2, 1]])
    return ret


def v(p, q, H):

    return np.array([
        H[0, p] * H[0, q],
        H[0, p] * H[1, q] + H[1, p] * H[0, q],
        H[1, p] * H[1, q],
        H[2, p] * H[0, q] + H[0, p] * H[2, q],
        H[2, p] * H[1, q] + H[1, p] * H[2, q],
        H[2, p] * H[2, q]
    ])


def get_camera_intrinsics(homographies):

    h_count = len(homographies)

    vec = []

    for i in range(0, h_count):
        curr = np.reshape(homographies[i], (3, 3))

        vec.append(v(0, 1, curr))
        vec.append(v(0, 0, curr) - v(1, 1, curr))

    vec = np.array(vec)

    b = np.linalg.lstsq(
        vec,
        np.zeros(h_count * 2),
    )[-1]

    w = b[0] * b[2] * b[5] - b[1]**2 * b[5] - b[0] * b[4]**2 + 2 * b[1] * b[3] * b[4] - b[2] * b[3]**2
    d = b[0] * b[2] - b[1]**2

    # if (d < 0):
    #     d = 0.01
    d = -d

    #
    alpha = np.sqrt(w / (d * b[0]))
    beta = np.sqrt(w / d**2 * b[0])
    gamma = np.sqrt(w / (d**2 * b[0])) * b[1]
    uc = (b[1] * b[4] - b[2] * b[3]) / d
    vc = (b[1] * b[3] - b[0] * b[4]) / d

    return np.array([
        [alpha, gamma, uc],
        [0,     beta,  vc],
        [0,     0,      1]
        ])


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
    homo_set = []
    for image in images:
        im = cv2.imread(images_path + "/" + image)
        homo_set.append(get_warped_circular(im))
        # warped_circular += (get_warped_circular(im))

    intrinsic = get_camera_intrinsics(homo_set)
    print(intrinsic)
    exit()

    warped_circular = np.array(warped_circular)

    conic = calculate_conic(warped_circular)
    # print(conic)
    a, s_mat, b = np.linalg.svd(conic)
    new_conic = np.matmul(a, s_mat, b)
    # print(new_conic)

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
    cv2.destroyAllWindows()