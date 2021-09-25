PART 1 (hough transform):
The hough transform is implemented using the Sobel algorithm to detect
the edges of an image. The algorithm determines the importance of the
points based on their gradient orientation and assign a constant value
of 1. There are 2 thresholds that are being used: magnitude threshold
(for gradient magnitude) and hough threshold for hough space.


PART 2 (Harris corner detection):
The Harris corner detection algorithm is implemented using a response
threshold and a non maximum suppression where the algorithm divides the
image into cells of MxN in which we assign a fix amount of features to be
detected in each cell. In addition, it applies a distance threshold between
features in the cell.


PART 3 (Feature Descriptor and Matching):
The SIFT-like descriptor is implemented using keypoints found by the Harris
corner detection algorithm. The algorithm computes the descriptor of each detected
keypoint using a 16*16 window and with a 4*4 sub window. In this algorithm, it uses
8 orientations (45 degrees) yielding a descriptor of 128 dimensions. Lastly, it uses
the ratio distance of the best and second best to perform matching with a threshold of
0.8.


PART 5A (SIFT Descriptor):

The SIFT descriptor is implemented using 4 octaves with s=3 resulting in 6 images/octave.
It uses a contrast threshold of 0.04 to filter the keypoints. Lastly, it applies SSD to
match keypoint.

NOTE: The algorithm is incomplete because the edge threshold wasn't apply to the keypoints.