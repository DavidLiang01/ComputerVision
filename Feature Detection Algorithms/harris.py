import cv2
import numpy as np
import math

img_c1 = cv2.imread("image_sets/contrast/contrast1.jpg")
img_c2 = cv2.imread("image_sets/contrast/contrast5.jpg")

img1 = cv2.imread("image_sets/yosemite/Yosemite1.jpg", 0)
img2 = cv2.imread("image_sets/yosemite/Yosemite2.jpg", 0)
img3 = cv2.imread("image_sets/contrast/contrast1.jpg", 0)
img4 = cv2.imread("image_sets/contrast/contrast5.jpg", 0)


# PART 2
def harris(img, window_size, r_threshold):
    ddepth = cv2.CV_32F
    ksize = 15
    row, col = img.shape
    alpha = 0.05
    keypoints = []
    dist_threshold = 20

    sub_row = int(row / 20)
    sub_col = int(col / 20)

    ix = cv2.Sobel(img, ddepth, 1, 0)
    iy = cv2.Sobel(img, ddepth, 0, 1)

    ix2 = np.square(ix)
    iy2 = np.square(iy)
    ixiy = np.multiply(ix, iy)

    ix2_gauss = cv2.GaussianBlur(ix2, (ksize, ksize), ksize / 3)
    iy2_gauss = cv2.GaussianBlur(iy2, (ksize, ksize), ksize / 3)
    ixiy_gauss = cv2.GaussianBlur(ixiy, (ksize, ksize), ksize / 3)

    r = np.zeros((row, col), dtype=np.float64)
    r_thres = np.zeros((row, col), dtype=np.float64)
    offset = int(window_size / 2)

    for y in range(offset, row - offset):
        for x in range(offset, col - offset):
            ix2_wind = ix2_gauss[y - offset:y + offset + 1, x - offset:x + offset + 1]
            iy2_wind = iy2_gauss[y - offset:y + offset + 1, x - offset:x + offset + 1]
            ixiy_wind = ixiy_gauss[y - offset:y + offset + 1, x - offset:x + offset + 1]

            ix2_sum = np.sum(ix2_wind)
            iy2_sum = np.sum(iy2_wind)
            ixiy_sum = np.sum(ixiy_wind)

            h = np.array([[ix2_sum, ixiy_sum], [ixiy_sum, iy2_sum]], np.float64)
            r[y, x] = np.linalg.det(h) - (alpha * (np.square(np.trace(h))))

            if r[y, x] > r_threshold:
                r_thres[y, x] = r[y, x]

    r_thres_sort = np.sort(r_thres, axis=None)[::-1]

    total_local_pixels = int(sub_col * sub_row)
    local_threshold = int(total_local_pixels * 0.01)
    global_threshold = r_thres_sort[int(len(r_thres_sort) * 0.2)]

    for y in range(sub_row, row + 1, sub_row):
        for x in range(sub_col, col + 1, sub_col):
            current_cell = r_thres[y - sub_row:y, x - sub_col:x]
            local_r_thres_sort = np.sort(current_cell, axis=None)[::-1]

            corner_count = 0
            for i in range(len(local_r_thres_sort)):

                if corner_count < local_threshold and len(local_r_thres_sort) != 0 \
                        and local_r_thres_sort[0] > global_threshold:

                    current_r1 = np.argwhere(current_cell == local_r_thres_sort[0])[0]
                    keypoints = np.append(keypoints,
                                          cv2.KeyPoint(float(current_r1[1] + x), float(current_r1[0] + y), 1))
                    corner_count += 1
                    idx_to_delete = np.array([])

                    for j in range(1, len(local_r_thres_sort)):
                        current_r2 = np.argwhere(current_cell == local_r_thres_sort[j])[0]

                        eucl_dist = np.linalg.norm(current_r1 - current_r2)

                        if eucl_dist < dist_threshold:
                            idx_to_delete = np.append(idx_to_delete, [j])

                    idx_to_delete = np.append(idx_to_delete, [0])
                    local_r_thres_sort = np.delete(local_r_thres_sort, idx_to_delete.astype(int))
                else:
                    break

    ix_img = cv2.convertScaleAbs(ix)
    iy_img = cv2.convertScaleAbs(iy)
    ixiy_img = cv2.convertScaleAbs(ixiy)
    r_img = cv2.convertScaleAbs(r)
    img_corners = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0),
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # cv2.imshow("ix", ix_img)
    # cv2.imshow("iy", iy_img)
    # cv2.imshow("ixiy", ixiy_img)
    # cv2.imshow("r", r_img)
    # cv2.imshow("harris corner", img_corners)
    # cv2.waitKey(0)

    return keypoints


window_size = 5
r_threshold = 800000


# keypoints1 = harris(img1, window_size, r_threshold)
# keypoints2 = harris(img2, window_size, r_threshold)

print("processing img 1")
keypoints1=harris(img3,window_size,r_threshold)
print("processing img 2")
keypoints2=harris(img4,window_size,r_threshold)
print("processing SIFT")



# PART 3
def compute_orien_histogram(cell):
    offset = 1
    orien_histogram = np.zeros(8, dtype=np.float64)

    for y in range(1, len(cell) - offset):
        for x in range(1, len(cell[0]) - offset):

            x_left = cell[y, x - offset]
            x_right = cell[y, x + offset]
            y_top = cell[y - offset, x]
            y_bottom = cell[y + offset, x]

            x_axis = x_right - x_left
            y_axis = y_bottom - y_top
            angle = np.arctan2(y_axis, x_axis) * 180 / math.pi
            magn = np.sqrt((x_axis ** 2) + (y_axis ** 2))

            if 0 <= angle <= 45:
                orien_histogram[0] += magn
            elif 45 < angle <= 90:
                orien_histogram[1] += magn
            elif 90 < angle <= 135:
                orien_histogram[2] += magn
            elif 135 < angle <= 180:
                orien_histogram[3] += magn
            elif -180 < angle < -135:
                orien_histogram[4] += magn
            elif 135 <= angle < -90:
                orien_histogram[5] += magn
            elif -90 <= angle < -45:
                orien_histogram[6] += magn
            elif -45 <= angle < 0:
                orien_histogram[7] += magn

    return orien_histogram


def sift_detector_harris(img_gray1, img_gray2, img_c1, img_c2, keypoints1, keypoints2):
    ssd_thres = 0.8
    matches = []
    images_descriptors = [[], []]

    xy_coord1 = cv2.KeyPoint_convert(keypoints1)
    xy_coord2 = cv2.KeyPoint_convert(keypoints2)
    images_coor = [xy_coord1, xy_coord2]

    img_gray1 = img_gray1.astype(np.int64)
    img_gray2 = img_gray2.astype(np.int64)
    images_gray = [img_gray1, img_gray2]

    for img_idx in range(len(images_coor)):
        for coor in images_coor[img_idx]:
            x, y = coor.astype(int)
            offset = 1
            current_key_descriptor = []

            cell = images_gray[img_idx][y - 7 - offset:y + 8 + offset + 1, x - 7 - offset:x + 8 + offset + 1]

            for row in range(4):
                for col in range(4):
                    sub_cell = cell[(row * 4):((row + 1) * 4) + offset + 1, (col * 4):((col + 1) * 4) + offset + 1]

                    current_key_descriptor = np.concatenate((current_key_descriptor, compute_orien_histogram(sub_cell)))

            norm_vector = np.linalg.norm(current_key_descriptor)

            if norm_vector != 0:
                norm_descriptor = current_key_descriptor / norm_vector
                norm_descriptor[norm_descriptor >= 0.2] = 0
                norm_descriptor = norm_descriptor / (np.linalg.norm(norm_descriptor))

                images_descriptors[img_idx].append(norm_descriptor)
            else:
                images_descriptors[img_idx].append(np.zeros(128))

    img1_descriptor, img2_descriptor = images_descriptors

    for img1_des_idx in range(len(img1_descriptor)):
        dist = []
        current_img1_des = img1_descriptor[img1_des_idx]

        for img2_des_idx in range(len(img2_descriptor)):
            current_img2_des = img2_descriptor[img2_des_idx]

            dist.append(np.sum((current_img1_des - current_img2_des) ** 2))

        dist_sort = np.sort(dist)

        ssd_best2 = dist_sort[1]
        if ssd_best2 != 0:
            ssd = dist_sort[0] / ssd_best2

            if ssd < ssd_thres:
                train_idx = np.argwhere(dist == dist_sort[0])[0]
                matches = np.append(matches, cv2.DMatch(int(img1_des_idx), int(train_idx), None, ssd))

    img_match = cv2.drawMatches(img_c1, keypoints1, img_c2, keypoints2, matches, None)
    img_corners1 = cv2.drawKeypoints(img_c1, keypoints1, None, color=(0, 255, 0),
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_corners2 = cv2.drawKeypoints(img_c2, keypoints2, None, color=(0, 255, 0),
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow("img1", img_corners1)
    cv2.imshow("img2", img_corners2)
    cv2.namedWindow("img_matches", cv2.WINDOW_NORMAL)
    cv2.imshow("img_matches", img_match)
    cv2.waitKey(0)


sift_detector_harris(img3, img4, img_c1, img_c2, keypoints1, keypoints2)


# PART 5

def create_dog_octave(img, num_images):
    ksize = 49
    sigma_init = ksize / 3
    sigma = sigma_init
    s = num_images - 3
    gauss_img = [img]
    dog_img = []
    for i in range(1, num_images):
        gauss_img.append(cv2.GaussianBlur(img, (ksize, ksize), sigma))
        sigma = 2 ** ((i / s) * sigma_init)

    for i in range(len(gauss_img) - 1):
        dog_img.append(gauss_img[i] - gauss_img[i + 1])

    return dog_img[1:len(dog_img) - 1]


def find_keypoints(octave_img):
    keypoints = []
    ite = (len(octave_img) - 3) + 1
    for i in range(ite):

        low_img = octave_img[i]
        mid_img = octave_img[i + 1]
        top_img = octave_img[i + 2]

        r, c = low_img.shape
        offset = 1
        for y in range(1, r):
            for x in range(1, c):

                low_img_cell = low_img[y - offset:y + offset + 1, x - offset:x + offset + 1]
                mid_img_cell = mid_img[y - offset:y + offset + 1, x - offset:x + offset + 1]
                top_img_cell = top_img[y - offset:y + offset + 1, x - offset:x + offset + 1]

                value = mid_img_cell[1, 1]
                contrast_thres = 0.04
                edge_thres = 10
                num_octave = 3
                threshold = math.floor(0.5 * contrast_thres / num_octave * 255)
                if abs(value) > threshold and abs(value) > edge_thres:

                    is_max_mid = len(np.argwhere(mid_img_cell >= value)) == 1
                    is_max_top = len(np.argwhere(top_img_cell >= value)) == 0
                    is_max_low = len(np.argwhere(low_img_cell >= value)) == 0
                    is_min_mid = len(np.argwhere(mid_img_cell <= value)) == 1
                    is_min_top = len(np.argwhere(top_img_cell <= value)) == 0
                    is_min_low = len(np.argwhere(low_img_cell <= value)) == 0

                    if (is_max_top and is_max_low and is_max_mid) or (is_min_low and is_min_top and is_min_mid):
                        keypoints.append(cv2.KeyPoint(float(x), float(y), 1))

    return keypoints


def sift_detector(img1, img2, img_c1, img_c2):
    num_octave = 4
    num_images = 6
    keypoints = [[], []]
    img_g = [img1, img2]

    for img_idx in range(len(img_g)):
        img = img_g[img_idx]

        for i in range(num_octave):
            octave_images = create_dog_octave(img, num_images)
            keypoints[img_idx] = np.concatenate((keypoints[img_idx], find_keypoints(octave_images)))
            img = cv2.resize(cv2.pyrDown(img), (640, 480), interpolation=cv2.INTER_CUBIC)

    sift_detector_harris(img1, img2, img_c1, img_c2, keypoints[0], keypoints[1])


sift_detector(img1, img2, img_c1, img_c2)
