import cv2
import numpy as np
import math

img = cv2.imread("house.jpg")
img_gray = cv2.imread("house.jpg", 0)

# PART 1

# PART A,B,C
def downsampling(img, factor):
    if factor in [2, 4, 8, 16]:
        downsize_img = img[::factor, ::factor, :]

        cv2.imshow("img factor " + str(factor), downsize_img)
        cv2.waitKey(0)

        if factor == 16:
            img_nn = cv2.resize(downsize_img, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
            img_bi = cv2.resize(downsize_img, None, fx=10, fy=10, interpolation=cv2.INTER_LINEAR)
            img_cu = cv2.resize(downsize_img, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)
            cv2.imshow("nearest neighbor", img_nn)
            cv2.imshow("bilinear interpolation", img_bi)
            cv2.imshow("bicubic interpolation", img_cu)
            cv2.waitKey(0)
        return downsize_img
    else:
        print("Invalid downscaling factor")


resize_img = downsampling(img, 2)
downsampling(img, 4)
downsampling(img, 8)
downsampling(img, 16)


# PART 2

# A
def diagonal_shift_img(img):
    ddepth = -1
    kernel_size = 200
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size - 1, 0] = 1
    shift_img = cv2.filter2D(img, ddepth, kernel, borderType=cv2.BORDER_ISOLATED)
    cv2.imshow("original img", img)
    cv2.imshow("img_shift", shift_img)
    cv2.waitKey(0)


diagonal_shift_img(resize_img)


# B
def gaussian_filter(n, sigma):
    range = math.floor(n / 2)
    x_val = np.arange(-range, range + 1)
    y_val = np.flip(x_val)
    xx, yy = np.meshgrid(x_val, y_val)
    A = sigma * math.sqrt(2 * math.pi)
    denom = 2 * (sigma ** 2)
    numer = -((xx ** 2) + (yy ** 2))
    gauss_kernel = np.exp(numer / denom)
    normalize = np.sum(gauss_kernel)
    gauss_kernel /= normalize
    return gauss_kernel


def gaussian_img(img, n, sigma):
    ddepth = -1
    gauss_kernel = gaussian_filter(n, sigma)
    gauss_img = cv2.filter2D(img, ddepth, gauss_kernel)
    cv2.imshow("original img", img)
    cv2.imshow("gauss_img " + str(13) + " ksize", gauss_img)
    cv2.waitKey(0)


n = 13
gaussian_img(img, n, n / 3)


# C
def gauss_diff(img, sigma, k, n):
    ddepth = cv2.CV_64FC1

    gaussian_filter1 = gaussian_filter(n, sigma)
    gaussian_filter2 = gaussian_filter(n, k * sigma)
    diff_gauss = np.subtract(gaussian_filter1, gaussian_filter2)

    diff_gauss_img = cv2.filter2D(img, ddepth, diff_gauss)

    cv2.imshow("original img", img)
    cv2.imshow("diff_gauss_img", diff_gauss_img)

    cv2.waitKey(0)


n = 3
gauss_diff(img_gray, n / 3, 4, n)


# PART 3

# A
def sobel_x(img):
    ddepth = cv2.CV_32F
    sobel_kernel = np.array([[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]], dtype=np.float32)
    sobel_kernel /= 8
    sobel_img = cv2.filter2D(img, ddepth, sobel_kernel)
    sobel_uint8 = cv2.convertScaleAbs(sobel_img)

    cv2.imshow("sobel_img_x", sobel_uint8)
    cv2.waitKey(0)
    return sobel_img


def sobel_y(img):
    ddepth = cv2.CV_32F
    sobel_kernel = np.array([[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]], dtype=np.float32)
    sobel_kernel /= 8
    sobel_img = cv2.filter2D(img, ddepth, sobel_kernel)
    sobel_uint8 = cv2.convertScaleAbs(sobel_img)

    cv2.imshow("sobel_img_y", sobel_uint8)
    cv2.waitKey(0)
    return sobel_img


img_x = sobel_x(img_gray)
img_y = sobel_y(img_gray)


# B
def pixel_orientation(img_x, img_y):
    orientation = np.arctan2(img_y, img_x) * 180 / math.pi
    orientation_uint8 = cv2.convertScaleAbs(orientation)
    cv2.imshow("orientation", orientation_uint8)
    cv2.waitKey(0)
    return orientation_uint8


angle = pixel_orientation(img_x, img_y)


# C
def pixel_magnitude(img_x, img_y):
    i_x = img_x ** 2
    i_y = img_y ** 2
    magnitude = np.sqrt(i_x + i_y)
    magnitude_uint8 = cv2.convertScaleAbs(magnitude)
    cv2.imshow("magnitude", magnitude_uint8)
    cv2.waitKey(0)
    return magnitude_uint8


intensity = pixel_magnitude(img_x, img_y)


# D
def canny_img(img):
    edges = cv2.Canny(img, 150, 200)
    cv2.imshow("canny", edges)
    cv2.waitKey(0)


canny_img(img_gray)


# Part 4

# A
def non_max_suppression(intensity, angle):
    print("Non max suppression: processing image ...")
    row, col = intensity.shape
    sup_mat = np.zeros((row, col), dtype=np.float32)

    for i in range(1, row - 1):
        for j in range(1, col - 1):
            if (0 <= angle[i, j] < 25) or (155 <= angle[i, j] <= 180):
                q = intensity[i, j + 1]
                r = intensity[i, j - 1]
            elif (25 <= angle[i, j] < 65):
                q = intensity[i + 1, j - 1]
                r = intensity[i - 1, j + 1]
            elif (65 <= angle[i, j] < 115):
                q = intensity[i + 1, j]
                r = intensity[i - 1, j]
            elif (115 <= angle[i, j] < 155):
                q = intensity[i - 1, j - 1]
                r = intensity[i + 1, j + 1]

            if (intensity[i, j] >= q) and (intensity[i, j] >= r):
                sup_mat[i, j] = intensity[i, j]
            else:
                sup_mat[i, j] = 0

    sup_mat_8 = cv2.convertScaleAbs(sup_mat)
    print("Done processing")
    cv2.imshow("non max supression", sup_mat_8)
    cv2.waitKey(0)


non_max_suppression(intensity, angle)
