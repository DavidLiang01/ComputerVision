import cv2
import numpy as np

img1 = cv2.imread("hough/hough1.png", 0)
img2 = cv2.imread("hough/hough2.png", 0)


# Part 1

def hough(img, hough_threshold, magnitude_threshold):
    ddepth = cv2.CV_32F

    sobelx = cv2.Sobel(img, ddepth, 1, 0)
    sobely = cv2.Sobel(img, ddepth, 0, 1)

    magnitude = cv2.magnitude(sobelx, sobely)

    r, c = magnitude.shape
    d = int(np.sqrt(np.square(r) + np.square(c)))
    theta_axis = np.arange(180)
    rho_axis = np.arange(-d, d, 1)
    hough_space = np.zeros((len(rho_axis), len(theta_axis)), dtype=np.float64)

    for y in range(r):
        for x in range(c):
            if magnitude[y, x] > magnitude_threshold:
                for angle in range(180):
                    current_rho = (x * np.cos(np.deg2rad(angle))) + (y * np.sin(np.deg2rad(angle)))
                    rho_index = np.argmin(np.abs(rho_axis - current_rho))
                    hough_space[rho_index, angle] += 1

    img_c = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    row, col = hough_space.shape

    for y in range(row):
        for x in range(col):
            if hough_space[y, x] > hough_threshold:
                rho = rho_axis[y]
                theta = theta_axis[x]
                a = np.cos(np.deg2rad(theta))
                b = np.sin(np.deg2rad(theta))

                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                cv2.line(img_c, (x1, y1), (x2, y2), (0, 0, 255), 1, lineType=cv2.LINE_AA)

    hough_space_abs = cv2.convertScaleAbs(hough_space)
    cv2.imshow("img", hough_space_abs)
    cv2.imshow("img_lines", img_c)
    cv2.waitKey(0)


hough_threshold1 = 115
magnitude_threshold1 = 30
hough_threshold2 = 190
magnitude_threshold2 = 70
hough(img1, hough_threshold1, magnitude_threshold1)
hough(img2, hough_threshold2, magnitude_threshold2)
